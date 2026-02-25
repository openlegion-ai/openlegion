"""Persistent workspace on disk — the agent's long-term memory.

Layout:
  /data/workspace/
  ├── PROJECT.md      # Shared fleet context (mounted read-only from host)
  ├── AGENTS.md       # Operating instructions (loaded into system prompt)
  ├── SOUL.md         # Per-agent identity, personality, tone
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
is building, the current priority, and hard constraints. Identity
files (SOUL.md, AGENTS.md, USER.md) are per-agent.
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
        "You are woken periodically. On each heartbeat:\n"
        "1. Check the blackboard for tasks or signals from other agents\n"
        "2. Continue working toward your current goal\n"
        "3. Report what you worked on to the user via notify_user\n"
        "   (what you did, progress made, any blockers)\n"
        "\nRemember: the blackboard is for collaborating with other agents.\n"
        "Use notify_user to keep the user informed of your progress.\n"
    ),
}

_MAX_FILE_SIZE = 200_000

# Bootstrap capping — limits for system prompt injection
_MAX_BOOTSTRAP = 40_000
_MAX_SYSTEM = 6_000

# Permission keys surfaced to agents in SYSTEM.md and Runtime Context.
# Keep in sync with the introspect endpoint in src/host/server.py.
INTROSPECT_PERM_KEYS = (
    "blackboard_read", "blackboard_write", "can_message",
    "can_publish", "can_subscribe", "allowed_apis",
    "allowed_credentials",
)
_MAX_AGENTS = 8_000
_MAX_SOUL = 4_000
_MAX_USER = 4_000
_MAX_MEMORY = 16_000

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

    def __init__(self, workspace_dir: str = "/data/workspace", initial_instructions: str = ""):
        self.root = Path(workspace_dir)
        self._initial_instructions = initial_instructions
        self._ensure_scaffold()

    def _ensure_scaffold(self) -> None:
        """Create workspace directory and default files if they don't exist."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / self.DAILY_DIR).mkdir(exist_ok=True)
        (self.root / self.LEARNINGS_DIR).mkdir(exist_ok=True)
        for filename, default_content in _SCAFFOLD_FILES.items():
            path = self.root / filename
            if not path.exists():
                # Seed AGENTS.md from template instructions on first creation
                if filename == "AGENTS.md" and self._initial_instructions:
                    content = (
                        "# Agent Instructions\n\n"
                        + self._initial_instructions.strip()
                        + "\n"
                    )
                    path.write_text(content)
                else:
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

        # SYSTEM.md — generated architecture guide (static preamble + snapshot)
        system = self._read_file("SYSTEM.md")
        if system and system.strip():
            system = system.strip()
            if len(system) > _MAX_SYSTEM:
                system = system[:_MAX_SYSTEM] + "\n\n... (truncated)"
            parts.append(system)

        for filename, cap in caps.items():
            content = self._read_file(filename)
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

    # Files agents are allowed to update themselves
    AGENT_WRITABLE = frozenset({"HEARTBEAT.md", "USER.md"})
    _MAX_WRITABLE_SIZE = 32_000  # 32KB cap for agent-writable files
    _MAX_BACKUPS_PER_FILE = 20

    def update_file(self, filename: str, content: str) -> dict:
        """Write a workspace file with backup versioning.

        Only files in AGENT_WRITABLE can be updated by agents.
        Creates a timestamped backup before overwriting.
        Returns metadata about the write.
        """
        if filename not in self.AGENT_WRITABLE:
            allowed = ", ".join(sorted(self.AGENT_WRITABLE))
            return {"error": f"Cannot update {filename}. Agents can only update: {allowed}"}

        if len(content) > self._MAX_WRITABLE_SIZE:
            return {"error": f"Content too large ({len(content)} chars). Max is {self._MAX_WRITABLE_SIZE}."}

        path = (self.root / filename).resolve()
        # Defense-in-depth: ensure resolved path stays within workspace
        if not path.is_relative_to(self.root.resolve()):
            return {"error": f"Invalid filename: {filename}"}
        backup_dir = self.root / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Back up existing content
        if path.exists():
            old_content = path.read_text(errors="replace")
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
            backup_path = backup_dir / f"{filename}.{timestamp}.bak"
            backup_path.write_text(old_content)
            self._rotate_backups(backup_dir, filename)

        path.write_text(content)
        self.append_daily_log(f"Updated workspace file: {filename}")
        return {
            "filename": filename,
            "size": len(content),
            "updated": True,
        }

    def _rotate_backups(self, backup_dir: Path, filename: str) -> None:
        """Keep only the most recent backups per file."""
        pattern = f"{filename}.*.bak"
        backups = sorted(backup_dir.glob(pattern))
        if len(backups) > self._MAX_BACKUPS_PER_FILE:
            for old in backups[: len(backups) - self._MAX_BACKUPS_PER_FILE]:
                try:
                    old.unlink()
                except OSError:
                    pass

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
        except OSError as e:
            logger.debug("Failed to rotate learnings file %s: %s", path, e)


# ── SYSTEM.md generation ──────────────────────────────────────

# Static preamble: architecture concepts that change rarely.
# This is the "how you tick" knowledge that helps agents self-improve.
_SYSTEM_MD_PREAMBLE = """\
# System Architecture

You are an agent running inside an isolated container in the OpenLegion mesh.
This document describes how the system works so you can operate effectively.

## How Your World Works

**Mesh host** — A central coordinator at your MESH_URL. Every external action
you take goes through it: LLM calls, web requests, blackboard reads/writes,
pub/sub events. You have no direct internet access.

**Credential vault** — You never hold API keys. When you call an LLM or
external API, the mesh injects credentials server-side. You cannot leak
what you cannot see.

**Blackboard** — A shared key-value store (SQLite) for agent-to-agent
coordination. Keys are hierarchical (e.g. `context/market`, `tasks/alice`).
You need explicit read/write permissions per key pattern.

**Pub/Sub** — Event topics for broadcast signals (`research_complete`,
`deploy_ready`). Subscribe to topics you care about; publish when you
have something other agents should react to.

## How Your Execution Works

**Context window** — Your conversation history is sent to the LLM on every
turn. Longer history = more tokens = higher cost. The system trims old
exchanges when the window fills, but first flushes important facts to
MEMORY.md (write-then-compact). To reduce cost: be concise, avoid
unnecessary tool calls, and save important facts to memory early.

**Tool calls** — Each tool round costs a full LLM call (system prompt +
entire history re-sent). Batching multiple actions in one response is
cheaper than one action per turn. The system detects repeated identical
tool calls and will block you to prevent waste.

**Memory search** — Your workspace files are searched with BM25 keyword
matching. Good queries use specific, distinctive terms. Your SQLite memory
DB supports both keyword (FTS5) and semantic (vector) search if embeddings
are configured. Use `memory_save` for structured facts you'll need later.

**Budget** — Each agent has daily and monthly cost caps. When you exceed
your daily budget, LLM calls are blocked until the next day. You can check
your budget with the `introspect` tool. Expensive operations: long
conversations (large context), vision/screenshot tools, embedding calls.

**Common errors and what they mean:**
- 403 Forbidden = permission denied (check your blackboard_read/write patterns)
- 429 Too Many Requests = rate limit hit (back off and retry)
- Budget exceeded = daily/monthly cap reached (wait or ask user to adjust)
- Tool loop detected = you called the same tool with same args too many times

## Coordination Philosophy

- **Fleet model**: no boss agent. You coordinate through the blackboard
  and workflows, not through a chain of command.
- **Private by default**: your memory and workspace are yours alone.
  Promote facts to the blackboard only when another agent needs them.
- **User-facing vs agent-facing**: report to the user via chat or
  `notify_user`. The blackboard is for agent-to-agent data only.
"""


def generate_system_md(introspect_data: dict, agent_id: str) -> str:
    """Generate SYSTEM.md content: static preamble + initial snapshot.

    The preamble teaches agents *how the system works*. A compact snapshot
    of permissions and fleet is appended as initial context but the
    authoritative live data comes from the ``## Runtime Context`` block
    injected into the system prompt on each turn by ``AgentLoop``.

    Role strings from the fleet are sanitized to prevent prompt injection
    via malicious agent registration data.
    """
    from src.shared.utils import sanitize_for_prompt

    parts = [_SYSTEM_MD_PREAMBLE]

    # Compact snapshot — helps agents on first message before runtime
    # context kicks in.  Kept deliberately short to avoid duplication
    # with the live runtime context block.
    perms = introspect_data.get("permissions")
    if perms:
        lines = ["## Your Permissions (snapshot)\n"]
        for key in INTROSPECT_PERM_KEYS:
            patterns = perms.get(key, [])
            if isinstance(patterns, list) and patterns:
                lines.append(f"- **{key}**: {', '.join(str(p) for p in patterns)}")
        parts.append("\n".join(lines))

    fleet = introspect_data.get("fleet")
    if fleet:
        names = []
        for agent in fleet:
            aid = sanitize_for_prompt(str(agent.get("id", "?")))
            role = sanitize_for_prompt(str(agent.get("role", "")))
            # Truncate to prevent bloat from malicious registration
            aid = aid[:60]
            role = role[:80]
            marker = " (you)" if aid == agent_id else ""
            names.append(f"{aid}{marker}" + (f" ({role})" if role else ""))
        parts.append(f"## Fleet: {', '.join(names)}")

    content = "\n\n".join(parts)
    if len(content) > _MAX_SYSTEM:
        content = content[:_MAX_SYSTEM].rsplit("\n", 1)[0] + "\n\n... (truncated)"
    return content


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
