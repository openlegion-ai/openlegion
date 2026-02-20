"""Persistent workspace on disk — the agent's long-term memory.

Layout:
  /data/workspace/
  ├── PROJECT.md      # Shared fleet context (mounted read-only from host)
  ├── AGENTS.md       # Operating instructions (loaded into system prompt)
  ├── SOUL.md         # Identity, personality, tone
  ├── USER.md         # User context, preferences
  ├── MEMORY.md       # Curated long-term memory (auto + manual)
  ├── HEARTBEAT.md    # Autonomous task rules (checked on heartbeat)
  ├── memory/
  │   ├── 2026-02-18.md   # Today's session log
  │   └── 2026-02-17.md   # Yesterday's log
  └── learnings/
      ├── errors.md       # Tool/task failure log with context
      └── corrections.md  # User corrections and preferences

All files are plain Markdown. Human-readable, git-versionable.
PROJECT.md is shared across all agents — it defines what the fleet
is building, the current priority, and hard constraints. Each agent
reads it on session start for alignment without centralized control.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.shared.utils import setup_logging

logger = setup_logging("agent.workspace")

_SCAFFOLD_FILES: dict[str, str] = {
    "AGENTS.md": (
        "# Agent Instructions\n\n"
        "Add operating instructions here. This file is loaded into your system prompt.\n"
    ),
    "SOUL.md": (
        "# Identity\n\n"
        "Define personality, tone, and behavioral guidelines here.\n"
    ),
    "USER.md": (
        "# User Context\n\n"
        "Record user preferences, background, and important context here.\n"
    ),
    "MEMORY.md": (
        "# Long-Term Memory\n\n"
        "Curated facts and important information are stored here automatically.\n"
        "You can also edit this file directly.\n"
    ),
    "HEARTBEAT.md": (
        "# Heartbeat Rules\n\n"
        "Define what to check on each heartbeat. The agent reads this file\n"
        "periodically and takes action when rules are triggered.\n\n"
        "## Rules\n\n"
        "- Check for pending tasks on the blackboard\n"
        "- Review and summarize new information\n"
    ),
}

_MAX_FILE_SIZE = 200_000

# Bootstrap capping — limits for system prompt injection
_MAX_BOOTSTRAP = 40_000
_MAX_AGENTS = 8_000
_MAX_SOUL = 4_000
_MAX_USER = 4_000
_MAX_MEMORY = 16_000

# Project-level SOUL.md fallback (mounted into container by runtime)
_SOUL_FALLBACK_PATH = Path("/app/SOUL.md")
# First line of default scaffold — used to detect un-customized SOUL.md
_DEFAULT_SOUL_PREFIX = "# Identity\n\nDefine personality"


_CORRECTION_SIGNALS = frozenset({
    "no,", "no.", "wrong", "incorrect", "that's not", "that is not",
    "actually,", "i meant", "you should", "don't do", "do not do",
    "stop doing", "not what i", "i said", "please don't", "instead,",
    "correction:", "fix:", "not like that",
})

_MAX_LEARNINGS_SIZE = 50_000


class WorkspaceManager:
    """Reads and writes the agent's persistent workspace files."""

    PROMPT_FILES = ("PROJECT.md", "AGENTS.md", "SOUL.md", "USER.md")
    MEMORY_FILE = "MEMORY.md"
    HEARTBEAT_FILE = "HEARTBEAT.md"
    DAILY_DIR = "memory"
    LEARNINGS_DIR = "learnings"
    ERRORS_FILE = "learnings/errors.md"
    CORRECTIONS_FILE = "learnings/corrections.md"

    def __init__(self, workspace_dir: str = "/data/workspace"):
        self.root = Path(workspace_dir)
        self._ensure_scaffold()

    def _ensure_scaffold(self) -> None:
        """Create workspace directory and default files if they don't exist."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / self.DAILY_DIR).mkdir(exist_ok=True)
        (self.root / self.LEARNINGS_DIR).mkdir(exist_ok=True)
        for filename, default_content in _SCAFFOLD_FILES.items():
            path = self.root / filename
            if not path.exists():
                path.write_text(default_content)
                logger.info(f"Created {filename}")

    # ── Reading ──────────────────────────────────────────────

    def load_prompt_context(self) -> str:
        """Load PROJECT.md, AGENTS.md, SOUL.md, USER.md into the system prompt."""
        parts: list[str] = []
        for filename in self.PROMPT_FILES:
            content = self._read_file(filename)
            if content and content.strip():
                parts.append(content.strip())
        return "\n\n---\n\n".join(parts) if parts else ""

    def load_memory(self) -> str:
        """Load MEMORY.md content."""
        return self._read_file(self.MEMORY_FILE) or ""

    def load_daily_logs(self, days: int = 2) -> str:
        """Load today's + yesterday's daily logs (most recent first)."""
        parts: list[str] = []
        today = datetime.now(UTC).date()
        for offset in range(days):
            date = today - timedelta(days=offset)
            filename = f"{self.DAILY_DIR}/{date.isoformat()}.md"
            content = self._read_file(filename)
            if content and content.strip():
                parts.append(f"## Session Log: {date.isoformat()}\n\n{content.strip()}")
        return "\n\n".join(parts) if parts else ""

    def get_bootstrap_content(self) -> str:
        """Load workspace files for system prompt with per-file and total caps.

        Loads AGENTS.md, SOUL.md, USER.md, MEMORY.md with individual size
        limits. Appends truncation notice when a file is capped. Enforces
        a total cap across all files.

        Daily logs are NOT included — agents access them via memory_search.
        """
        caps = {
            "AGENTS.md": _MAX_AGENTS,
            "SOUL.md": _MAX_SOUL,
            "USER.md": _MAX_USER,
            "MEMORY.md": _MAX_MEMORY,
        }

        parts: list[str] = []

        # PROJECT.md has no individual cap but counts toward total
        project = self._read_file("PROJECT.md")
        if project and project.strip():
            parts.append(project.strip())

        for filename, cap in caps.items():
            content = self._read_file(filename)

            # SOUL.md: fall back to project-level if workspace copy is default scaffold
            if filename == "SOUL.md" and (
                not content or not content.strip()
                or content.strip().startswith(_DEFAULT_SOUL_PREFIX)
            ):
                fallback = self._read_external(str(_SOUL_FALLBACK_PATH))
                if fallback and fallback.strip():
                    content = fallback

            if not content or not content.strip():
                continue
            content = content.strip()
            if len(content) > cap:
                content = content[:cap] + (
                    "\n\n... (truncated, use memory_search for full content)"
                )
            parts.append(content)

        combined = "\n\n---\n\n".join(parts)
        if len(combined) > _MAX_BOOTSTRAP:
            combined = combined[:_MAX_BOOTSTRAP] + (
                "\n\n... (bootstrap truncated, use memory_search for full content)"
            )
        return combined

    def _read_file(self, relative_path: str) -> str | None:
        path = self.root / relative_path
        if not path.exists() or not path.is_file():
            return None
        try:
            return path.read_text(errors="replace")[:_MAX_FILE_SIZE]
        except Exception as e:
            logger.warning(f"Failed to read {relative_path}: {e}")
            return None

    @staticmethod
    def _read_external(absolute_path: str) -> str | None:
        """Read a file by absolute path (e.g. project-level mounts)."""
        p = Path(absolute_path)
        if not p.exists() or not p.is_file():
            return None
        try:
            return p.read_text(errors="replace")[:_MAX_FILE_SIZE]
        except Exception:
            return None

    # ── Writing ──────────────────────────────────────────────

    def append_daily_log(self, entry: str) -> None:
        """Append an entry to today's daily log file."""
        today = datetime.now(UTC).date().isoformat()
        path = self.root / self.DAILY_DIR / f"{today}.md"
        timestamp = datetime.now(UTC).strftime("%H:%M")
        line = f"- [{timestamp}] {entry}\n"
        with path.open("a") as f:
            f.write(line)

    def append_memory(self, content: str) -> None:
        """Append to MEMORY.md (used by context manager before compacting)."""
        path = self.root / self.MEMORY_FILE
        with path.open("a") as f:
            f.write(f"\n{content}\n")

    # ── Search ───────────────────────────────────────────────

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """BM25 keyword search over all markdown files in workspace.

        Returns list of {"file": str, "snippet": str, "score": float}.
        """
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        documents: list[tuple[str, list[str], str]] = []
        for md_file in sorted(self.root.rglob("*.md")):
            content = md_file.read_text(errors="replace")[:_MAX_FILE_SIZE]
            if not content.strip():
                continue
            rel = str(md_file.relative_to(self.root))
            tokens = _tokenize(content)
            documents.append((rel, tokens, content))

        if not documents:
            return []

        scores = _bm25_score(query_terms, documents)

        ranked = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, (rel, _tokens, content) in ranked[:max_results]:
            if score <= 0:
                break
            snippet = _extract_snippet(content, query_terms, max_len=300)
            results.append({"file": rel, "snippet": snippet, "score": round(score, 3)})

        return results

    def list_memory_files(self) -> list[str]:
        """List all daily log files, most recent first."""
        daily_dir = self.root / self.DAILY_DIR
        if not daily_dir.exists():
            return []
        files = sorted(daily_dir.glob("*.md"), reverse=True)
        return [str(f.relative_to(self.root)) for f in files]

    # ── Learnings ────────────────────────────────────────────────

    def record_error(self, tool: str, error: str, context: str = "") -> None:
        """Record a tool or task failure for future avoidance."""
        path = self.root / self.ERRORS_FILE
        self._rotate_if_large(path)
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
        entry = f"- [{timestamp}] **{tool}**: {error}"
        if context:
            entry += f"\n  Context: {context}"
        with path.open("a") as f:
            f.write(entry + "\n")

    def record_correction(self, original: str, correction: str) -> None:
        """Record a user correction for learning."""
        path = self.root / self.CORRECTIONS_FILE
        self._rotate_if_large(path)
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
        entry = (
            f"- [{timestamp}] Original: {original[:200]}\n"
            f"  Correction: {correction[:500]}\n"
        )
        with path.open("a") as f:
            f.write(entry)

    @staticmethod
    def looks_like_correction(message: str) -> bool:
        """Heuristic: does this user message look like a correction?"""
        lower = message.lower().strip()
        return any(lower.startswith(s) for s in _CORRECTION_SIGNALS)

    def get_learnings_context(self, max_chars: int = 3000) -> str:
        """Load recent errors and corrections for system prompt injection."""
        parts: list[str] = []
        for relpath, heading in [
            (self.ERRORS_FILE, "## Recent Errors (avoid repeating)"),
            (self.CORRECTIONS_FILE, "## User Corrections (follow these)"),
        ]:
            content = self._read_file(relpath)
            if content and content.strip():
                lines = content.strip().splitlines()
                tail = "\n".join(lines[-20:])
                parts.append(f"{heading}\n\n{tail}")
        combined = "\n\n".join(parts)
        return combined[:max_chars] if combined else ""

    def load_heartbeat_rules(self) -> str:
        """Load HEARTBEAT.md content for autonomous operation."""
        return self._read_file(self.HEARTBEAT_FILE) or ""

    def _rotate_if_large(self, path: Path) -> None:
        """Trim old entries when a learning file exceeds the size limit."""
        if not path.exists():
            return
        try:
            size = path.stat().st_size
            if size > _MAX_LEARNINGS_SIZE:
                lines = path.read_text(errors="replace").splitlines()
                half = len(lines) // 2
                path.write_text("\n".join(lines[half:]) + "\n")
        except OSError:
            pass


# ── BM25 implementation (no external deps) ───────────────────

_STOP_WORDS = frozenset(
    "a an and are as at be by for from has have i in is it of on or that the to was with".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _bm25_score(
    query_terms: list[str],
    documents: list[tuple[str, list[str], str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Compute BM25 scores for query against documents.

    documents is list of (filename, tokens, raw_content).
    """
    n = len(documents)
    avg_dl = sum(len(tokens) for _, tokens, _ in documents) / n if n else 1

    # Document frequency for each term
    df: dict[str, int] = {}
    for _, tokens, _ in documents:
        seen = set(tokens)
        for term in seen:
            df[term] = df.get(term, 0) + 1

    scores: list[float] = []
    for _, tokens, _ in documents:
        dl = len(tokens)
        tf_map: dict[str, int] = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        score = 0.0
        for term in query_terms:
            if term not in df:
                continue
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            tf = tf_map.get(term, 0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avg_dl)
            score += idf * numerator / denominator

        scores.append(score)

    return scores


def _extract_snippet(content: str, query_terms: list[str], max_len: int = 300) -> str:
    """Extract the most relevant snippet from content around query terms."""
    lines = content.splitlines()
    best_line = ""
    best_score = -1

    for line in lines:
        line_lower = line.lower()
        score = sum(1 for term in query_terms if term in line_lower)
        if score > best_score:
            best_score = score
            best_line = line

    if not best_line and lines:
        best_line = lines[0]

    return best_line[:max_len].strip()
