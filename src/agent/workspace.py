"""Persistent workspace on disk — the agent's long-term memory.

Layout:
  /data/workspace/
  ├── TEAM.md           # Shared fleet context (mounted read-only from host)
  ├── INSTRUCTIONS.md   # Operating procedures, workflow rules, domain knowledge
  ├── SOUL.md           # Per-agent identity, personality, tone
  ├── USER.md           # User context, preferences
  ├── MEMORY.md         # Curated long-term memory (auto + manual)
  ├── HEARTBEAT.md      # Autonomous task rules (checked on heartbeat)
  ├── INTERFACE.md      # Public collaboration contract (read by other agents)
  ├── memory/
  │   ├── 2026-02-18.md   # Today's session log
  │   └── 2026-02-17.md   # Yesterday's log
  └── learnings/
      ├── errors.md       # Tool/task failure log with context
      └── corrections.md  # User corrections and preferences

All files are plain Markdown. Human-readable, git-versionable.
TEAM.md is shared across all team members — it defines what the team
is building, the current priority, and hard constraints. Identity
files (SOUL.md, INSTRUCTIONS.md, USER.md) are per-agent.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.shared.paths import resolve_under_root
from src.shared.utils import dumps_safe, sanitize_for_prompt, setup_logging

logger = setup_logging("agent.workspace")

_SCAFFOLD_FILES: dict[str, str] = {
    "INSTRUCTIONS.md": "# Instructions\n",
    "SOUL.md": "# Identity\n",
    "USER.md": "# User Context\n",
    "MEMORY.md": "# Long-Term Memory\n",
    "INTERFACE.md": "# Interface\n",
}
# HEARTBEAT.md is intentionally NOT bootstrapped. Most agents never write
# heartbeat rules — leaving the file absent (vs. empty) keeps the workspace
# directory clean and avoids signalling "this agent has rules" when it
# doesn't. Readers (loop._is_heartbeat_empty, workspace.load_heartbeat_rules,
# /heartbeat-context, /workspace listing) all treat missing identically to
# empty. The file is created lazily on first write — via either
# ``edit_agent(field="heartbeat")`` (mesh-side PUT /workspace/HEARTBEAT.md),
# the ``update_workspace`` agent tool, or the dashboard editor. Templates
# that ship explicit heartbeat rules still seed the file via
# ``initial_heartbeat`` below.

_MAX_FILE_SIZE = 200_000

# Bootstrap capping — limits for system prompt injection
_MAX_BOOTSTRAP = 48_000
_MAX_SYSTEM = 6_000

# Headers prepended to workspace files in the system prompt so the LLM
# knows what each block is. Skipped when the file already starts with
# its own markdown heading (``# ...``).
_BOOTSTRAP_HEADERS: dict[str, str] = {
    "TEAM.md": "Fleet-Wide Context",
    "SYSTEM.md": "System Architecture",
    "INSTRUCTIONS.md": "Your Operating Procedures & Domain Knowledge",
    "SOUL.md": "Your Identity & Personality",
    "USER.md": "User Preferences & Corrections",
    "MEMORY.md": "Long-Term Memory (auto-curated)",
}


def _maybe_add_header(filename: str, content: str) -> str:
    """Prepend a descriptive header to a workspace file if it lacks its own."""
    desc = _BOOTSTRAP_HEADERS.get(filename, "")
    if not desc or content.lstrip().startswith("#"):
        return content
    return f"## {filename} — {desc}\n\n{content}"

# Permission keys surfaced to agents in SYSTEM.md and Runtime Context.
# Keep in sync with the introspect endpoint in src/host/server.py.
INTROSPECT_PERM_KEYS = (
    "blackboard_read", "blackboard_write", "can_message",
    "can_publish", "can_subscribe", "allowed_apis",
    "allowed_credentials",
)
_MAX_INSTRUCTIONS = 12_000
_MAX_SOUL = 4_000
_MAX_USER = 4_000
_MAX_MEMORY = 16_000

# Public mapping for external consumers (skill response, dashboard).
# Files not listed have no per-file bootstrap cap.
BOOTSTRAP_CAPS: dict[str, int] = {
    "INSTRUCTIONS.md": _MAX_INSTRUCTIONS,
    "SOUL.md": _MAX_SOUL,
    "USER.md": _MAX_USER,
    "MEMORY.md": _MAX_MEMORY,
}

_CORRECTION_SIGNALS = frozenset({
    "no,", "no.", "wrong", "incorrect", "that's not", "that is not",
    "actually,", "i meant", "you should", "don't do", "do not do",
    "stop doing", "not what i", "i said", "please don't", "instead,",
    "correction:", "fix:", "not like that",
})

_MAX_LEARNINGS_SIZE = 50_000


# ── INTERFACE.md → structured capabilities derivation ─────────
#
# Task 8 introduces structured routing fields on AgentConfig
# (capabilities / preferred_inputs / expected_outputs / escalation_to /
# forbidden). For agents created before Task 8 the structured fields are
# absent; their INTERFACE.md may already declare the same information
# free-form via headings. ``_derive_capabilities_from_interface`` parses
# those headings and returns a dict shaped for back-fill into agents.yaml.
# Best-effort + idempotent: it is intended to run once when
# ``capabilities`` is missing/empty AND INTERFACE.md is on disk; the
# caller persists the result and the structured field becomes the source
# of truth thereafter.

_HEADING_RE = re.compile(r"^\s*##+\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$")

# Heading aliases — case-insensitive. Same header may have alternate
# phrasings across templates.
_INTERFACE_SECTIONS: dict[str, tuple[str, ...]] = {
    "capabilities": ("capabilities", "what i do", "role"),
    "preferred_inputs": ("preferred inputs", "accepts", "inputs"),
    "expected_outputs": ("expected outputs", "produces", "outputs"),
    "forbidden": ("forbidden", "do not", "refuses", "never"),
}
_ESCALATION_HEADINGS: tuple[str, ...] = ("escalation", "escalate to", "escalation to")


def _empty_interface_fields() -> dict[str, list[str] | str | None]:
    """Default-empty shape for the derived structured fields."""
    return {
        "capabilities": [],
        "preferred_inputs": [],
        "expected_outputs": [],
        "escalation_to": None,
        "forbidden": [],
    }


def _parse_interface_text(content: str) -> dict[str, list[str] | str | None]:
    """Parse INTERFACE.md text into structured routing fields.

    Walks markdown headings; for each known section, slurps following
    bullets up to the next heading. Headings are matched
    case-insensitively against the alias table. Returns the same shape
    as :func:`_derive_capabilities_from_interface` — empty defaults
    when no matching headings are present.
    """
    if not content:
        return _empty_interface_fields()

    current: str | None = None
    sections: dict[str, list[str]] = {k: [] for k in _INTERFACE_SECTIONS}
    escalation_lines: list[str] = []

    for raw_line in content.splitlines():
        m = _HEADING_RE.match(raw_line)
        if m:
            heading = m.group(1).strip().lower()
            current = None
            for key, aliases in _INTERFACE_SECTIONS.items():
                if heading in aliases:
                    current = key
                    break
            else:
                if heading in _ESCALATION_HEADINGS:
                    current = "__escalation__"
            continue

        if current is None:
            continue

        bullet = _BULLET_RE.match(raw_line)
        if not bullet:
            continue
        item = bullet.group(1).strip()
        if not item:
            continue
        if current == "__escalation__":
            escalation_lines.append(item)
        else:
            sections[current].append(item)

    return {
        "capabilities": sections["capabilities"],
        "preferred_inputs": sections["preferred_inputs"],
        "expected_outputs": sections["expected_outputs"],
        "forbidden": sections["forbidden"],
        "escalation_to": escalation_lines[0] if escalation_lines else None,
    }


def _derive_capabilities_from_interface(
    workspace_path: str | Path,
) -> dict[str, list[str] | str | None]:
    """Parse INTERFACE.md headings into structured routing fields.

    Returns a dict with the keys ``capabilities``, ``preferred_inputs``,
    ``expected_outputs``, ``forbidden`` (lists) and ``escalation_to``
    (string or ``None``). Missing sections produce empty defaults.

    Best-effort: a missing file, unreadable file, or unparseable content
    yields the empty defaults — never raises. Intended to run once on
    first read; callers persist the result back to agents.yaml so the
    derivation is not repeated.

    The parser recognises the standard section headings used by the
    Task-8 templates (``## Capabilities``, ``## Accepts``, ``## Produces``,
    ``## Escalation``, ``## Forbidden``) and a few common synonyms used by
    pre-Task-8 templates (``## Role`` → capabilities, ``## Inputs`` →
    preferred_inputs, etc.).
    """
    path = Path(workspace_path) / "INTERFACE.md"
    if not path.exists() or not path.is_file():
        return _empty_interface_fields()

    try:
        content = path.read_text(errors="replace")
    except Exception:
        return _empty_interface_fields()

    return _parse_interface_text(content)


class WorkspaceManager:
    """Reads and writes the agent's persistent workspace files."""

    MEMORY_FILE = "MEMORY.md"
    HEARTBEAT_FILE = "HEARTBEAT.md"
    DAILY_DIR = "memory"
    LEARNINGS_DIR = "learnings"
    ERRORS_FILE = "learnings/errors.md"
    CORRECTIONS_FILE = "learnings/corrections.md"
    CHAT_TRANSCRIPT = "chat_transcript.jsonl"
    CHAT_ARCHIVE_DIR = "chat_archive"
    ACTIVITY_LOG = "activity.jsonl"

    def __init__(
        self,
        workspace_dir: str = "/data/workspace",
        initial_instructions: str = "",
        initial_soul: str = "",
        initial_heartbeat: str = "",
        initial_interface: str = "",
    ):
        self.root = Path(workspace_dir)
        self._initial_instructions = initial_instructions
        self._initial_soul = initial_soul
        self._initial_heartbeat = initial_heartbeat
        self._initial_interface = initial_interface
        self._ensure_scaffold()

        # Caches — invalidated by mtime changes or explicit writes
        self._bootstrap_cache: str | None = None
        self._bootstrap_mtimes: dict[str, float] = {}
        self._learnings_cache: str | None = None
        self._learnings_mtimes: dict[str, float] = {}

    def _ensure_scaffold(self) -> None:
        """Create workspace directory and default files if they don't exist."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / self.DAILY_DIR).mkdir(exist_ok=True)
        (self.root / self.LEARNINGS_DIR).mkdir(exist_ok=True)

        # Migration: AGENTS.md → INSTRUCTIONS.md
        old_agents = self.root / "AGENTS.md"
        new_instructions = self.root / "INSTRUCTIONS.md"
        if old_agents.exists() and not new_instructions.exists():
            old_agents.rename(new_instructions)
            logger.info("Migrated AGENTS.md → INSTRUCTIONS.md")

        # Migration: mark existing instructions as playbook-aware.
        # Only appends the sentinel — does NOT rewrite content, preserving
        # any user customizations made via the dashboard.
        instructions_file = self.root / "INSTRUCTIONS.md"
        if (
            instructions_file.exists()
            and self._initial_instructions
            and "<!-- playbook_v2 -->" in self._initial_instructions
            and "<!-- playbook_v2 -->" not in instructions_file.read_text(errors="replace")
        ):
            with open(instructions_file, "a") as f:
                f.write("\n<!-- playbook_v2 -->\n")
            logger.info("Marked operator instructions as playbook-aware (v2)")

        for filename, default_content in _SCAFFOLD_FILES.items():
            path = self.root / filename
            if not path.exists():
                # Seed from template content on first creation
                if filename == "INSTRUCTIONS.md" and self._initial_instructions:
                    content = (
                        "# Instructions\n\n"
                        + self._initial_instructions.strip()
                        + "\n"
                    )
                    path.write_text(content)
                elif filename == "SOUL.md" and self._initial_soul:
                    content = (
                        "# Identity\n\n"
                        + self._initial_soul.strip()
                        + "\n"
                    )
                    path.write_text(content)
                elif filename == "INTERFACE.md" and self._initial_interface:
                    content = (
                        "# Interface\n\n"
                        + self._initial_interface.strip()
                        + "\n"
                    )
                    path.write_text(content)
                else:
                    path.write_text(default_content)
                logger.info(f"Created {filename}")

        # HEARTBEAT.md: lazy-create. Only seed when a template supplied
        # explicit rules; otherwise leave the file absent until first edit.
        # See the _SCAFFOLD_FILES comment for the rationale.
        heartbeat_path = self.root / self.HEARTBEAT_FILE
        if not heartbeat_path.exists() and self._initial_heartbeat:
            heartbeat_path.write_text(
                "# Heartbeat Rules\n\n"
                + self._initial_heartbeat.strip()
                + "\n"
            )
            logger.info("Created %s (seeded from template)", self.HEARTBEAT_FILE)

        # Idempotent heartbeat refresh: mirrors the INSTRUCTIONS.md
        # playbook_v2 sentinel pattern above. When the template carries a
        # versioned marker (latest: ``heartbeat_v3_rate_delivery``) and
        # the live file lacks the LATEST one, overwrite from the
        # template. Versioned markers let us roll system-managed
        # heartbeat updates forward without touching user-customised
        # heartbeats (which won't have any marker because they replaced
        # the template). New markers added in future revisions extend
        # ``HEARTBEAT_SENTINELS`` at the end — the last entry is the
        # current expectation; earlier entries stay as evidence of
        # prior migrations.
        from src.shared.types import HEARTBEAT_SENTINELS
        latest_sentinel = HEARTBEAT_SENTINELS[-1] if HEARTBEAT_SENTINELS else None
        if (
            heartbeat_path.exists()
            and self._initial_heartbeat
            and latest_sentinel
            and f"<!-- {latest_sentinel} -->" in self._initial_heartbeat
        ):
            try:
                existing = heartbeat_path.read_text(errors="replace")
            except Exception:
                existing = ""
            # Refresh ONLY when the file carries at least one prior
            # sentinel (proving it's a system-managed heartbeat we
            # have rights to roll forward) AND lacks the latest one.
            # Skipping the "any old sentinel present" check would
            # silently overwrite user-customised heartbeats — those
            # don't carry ANY marker because the user replaced the
            # template — every time we ship a new sentinel bump.
            has_any_old_sentinel = any(
                f"<!-- {s} -->" in existing for s in HEARTBEAT_SENTINELS
            )
            needs_refresh = (
                has_any_old_sentinel
                and f"<!-- {latest_sentinel} -->" not in existing
            )
            if needs_refresh:
                heartbeat_path.write_text(
                    "# Heartbeat Rules\n\n"
                    + self._initial_heartbeat.strip()
                    + "\n"
                )
                logger.info(
                    "Refreshed %s from versioned template",
                    self.HEARTBEAT_FILE,
                )

    # ── Reading ──────────────────────────────────────────────

    def load_memory(self) -> str:
        """Load MEMORY.md content."""
        return self._read_file(self.MEMORY_FILE) or ""

    def load_daily_logs(self, days: int = 2) -> str:
        """Load today's + yesterday's daily logs (most recent first)."""
        parts: list[str] = []
        today = datetime.now(timezone.utc).date()
        for offset in range(days):
            date = today - timedelta(days=offset)
            filename = f"{self.DAILY_DIR}/{date.isoformat()}.md"
            content = self._read_file(filename)
            if content and content.strip():
                parts.append(f"## Session Log: {date.isoformat()}\n\n{content.strip()}")
        return "\n\n".join(parts) if parts else ""

    # Bootstrap files searched in order. ``TEAM.md`` / ``team.md`` are
    # the canonical names — ``team.md`` is the dashboard-pushed variant.
    _BOOTSTRAP_FILES = (
        "TEAM.md", "team.md",
        "SYSTEM.md", "INSTRUCTIONS.md", "SOUL.md", "USER.md", "MEMORY.md",
    )

    @staticmethod
    def _get_mtime(path: Path) -> float:
        """Get file mtime, returning 0.0 if the file doesn't exist."""
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def _check_mtimes(self, filenames: tuple | list, cached_mtimes: dict) -> bool:
        """Return True if any file's mtime differs from the cached value."""
        return any(cached_mtimes.get(filename) != self._get_mtime(self.root / filename) for filename in filenames)

    def _snapshot_mtimes(self, filenames: tuple | list, target: dict) -> None:
        """Record current mtimes for the given filenames."""
        for filename in filenames:
            target[filename] = self._get_mtime(self.root / filename)

    def invalidate_bootstrap_cache(self) -> None:
        """Explicitly invalidate the bootstrap cache."""
        self._bootstrap_cache = None

    def get_bootstrap_content(self) -> str:
        """Load workspace files for system prompt with per-file and total caps.

        Loads INSTRUCTIONS.md, SOUL.md, USER.md, MEMORY.md with individual
        size limits. Appends truncation notice when a file is capped. Enforces
        a total cap across all files.

        Daily logs are NOT included — agents access them via memory_search.

        Results are cached with mtime-based invalidation and pre-sanitized.
        """
        if self._bootstrap_cache is not None and not self._check_mtimes(
            self._BOOTSTRAP_FILES, self._bootstrap_mtimes,
        ):
            return self._bootstrap_cache

        caps = {
            "INSTRUCTIONS.md": _MAX_INSTRUCTIONS,
            "SOUL.md": _MAX_SOUL,
            "USER.md": _MAX_USER,
            "MEMORY.md": _MAX_MEMORY,
        }

        parts: list[str] = []

        # TEAM.md (canonical) / team.md (dashboard-pushed) come first.
        team = self._read_file("TEAM.md") or self._read_file("team.md")
        if team and team.strip():
            parts.append(_maybe_add_header("TEAM.md", team.strip()))
        # Note: missing TEAM.md is normal for solo agents (not in a
        # team). No warning — the dashboard pushes TEAM.md only for
        # agents assigned to a team.

        # SYSTEM.md — generated architecture guide (static preamble + snapshot)
        system = self._read_file("SYSTEM.md")
        if system and system.strip():
            system = system.strip()
            if len(system) > _MAX_SYSTEM:
                system = system[:_MAX_SYSTEM] + "\n\n... (truncated)"
            parts.append(_maybe_add_header("SYSTEM.md", system))

        for filename, cap in caps.items():
            content = self._read_file(filename)
            if not content or not content.strip():
                continue
            content = content.strip()
            if len(content) > cap:
                content = content[:cap] + (
                    "\n\n... (truncated, use memory_search for full content)"
                )
            parts.append(_maybe_add_header(filename, content))

        combined = "\n\n---\n\n".join(parts)
        if len(combined) > _MAX_BOOTSTRAP:
            combined = combined[:_MAX_BOOTSTRAP] + (
                "\n\n... (bootstrap truncated, use memory_search for full content)"
            )
        # Pre-sanitize so callers never need to re-sanitize unchanged content
        combined = sanitize_for_prompt(combined)

        self._snapshot_mtimes(self._BOOTSTRAP_FILES, self._bootstrap_mtimes)
        self._bootstrap_cache = combined
        return combined

    def _read_file(self, relative_path: str) -> str | None:
        path = resolve_under_root(self.root, relative_path)
        if path is None:
            return None
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
        today = datetime.now(timezone.utc).date().isoformat()
        path = self.root / self.DAILY_DIR / f"{today}.md"
        timestamp = datetime.now(timezone.utc).strftime("%H:%M")
        line = f"- [{timestamp}] {entry}\n"
        with path.open("a") as f:
            f.write(line)

    def append_memory(self, content: str) -> None:
        """Append to MEMORY.md (used by context manager before compacting)."""
        path = self.root / self.MEMORY_FILE
        with path.open("a") as f:
            f.write(f"\n{content}\n")
        self._bootstrap_cache = None  # MEMORY.md is part of bootstrap

    # Files agents are allowed to update themselves
    AGENT_WRITABLE = frozenset({"HEARTBEAT.md", "USER.md", "SOUL.md", "INSTRUCTIONS.md", "INTERFACE.md"})
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

        # Defense-in-depth: ensure resolved path stays within workspace
        path = resolve_under_root(self.root, filename)
        if path is None:
            return {"error": f"Invalid filename: {filename}"}
        backup_dir = self.root / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Back up existing content
        if path.exists():
            old_content = path.read_text(errors="replace")
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            backup_path = backup_dir / f"{filename}.{timestamp}.bak"
            backup_path.write_text(old_content)
            self._rotate_backups(backup_dir, filename)

        path.write_text(content)
        self._bootstrap_cache = None  # invalidate — file changed
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
                with contextlib.suppress(OSError):
                    old.unlink()

    # ── Search ───────────────────────────────────────────────

    def search(
        self, query: str, max_results: int = 5,
        exclude_files: set[str] | None = None,
    ) -> list[dict]:
        """BM25 keyword search over all markdown files in workspace.

        *exclude_files* — optional set of filenames (relative to workspace
        root, e.g. ``{"MEMORY.md", "SOUL.md"}``) to skip.  Useful when the
        caller has already injected those files (e.g. bootstrap content).

        Returns list of {"file": str, "snippet": str, "score": float}.
        """
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        documents: list[tuple[str, list[str], str]] = []
        for md_file in sorted(self.root.rglob("*.md")):
            rel = str(md_file.relative_to(self.root))
            if exclude_files and rel in exclude_files:
                continue
            content = md_file.read_text(errors="replace")[:_MAX_FILE_SIZE]
            if not content.strip():
                continue
            tokens = _tokenize(content)
            documents.append((rel, tokens, content))

        if not documents:
            return []

        scores = _bm25_score(query_terms, documents)

        ranked = sorted(
            zip(scores, documents, strict=True),
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
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        entry = f"- [{timestamp}] **{tool}**: {error}"
        if context:
            entry += f"\n  Context: {context}"
        with path.open("a") as f:
            f.write(entry + "\n")
        self._learnings_cache = None

    def record_correction(self, original: str, correction: str) -> None:
        """Record a user correction for learning."""
        path = self.root / self.CORRECTIONS_FILE
        self._rotate_if_large(path)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        entry = (
            f"- [{timestamp}] Original: {original[:200]}\n"
            f"  Correction: {correction[:500]}\n"
        )
        with path.open("a") as f:
            f.write(entry)
        self._learnings_cache = None

    @staticmethod
    def looks_like_correction(message: str) -> bool:
        """Heuristic: does this user message look like a correction?"""
        lower = message.lower().strip()
        return any(lower.startswith(s) for s in _CORRECTION_SIGNALS)

    _LEARNINGS_FILES = (ERRORS_FILE, CORRECTIONS_FILE)

    def get_learnings_context(self, max_chars: int = 3000) -> str:
        """Load recent errors and corrections for system prompt injection.

        Results are cached with mtime-based invalidation and pre-sanitized.
        """
        if self._learnings_cache is not None and not self._check_mtimes(
            self._LEARNINGS_FILES, self._learnings_mtimes,
        ):
            return self._learnings_cache

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
        combined = combined[:max_chars] if combined else ""

        if combined:
            combined = sanitize_for_prompt(combined)

        self._snapshot_mtimes(self._LEARNINGS_FILES, self._learnings_mtimes)
        self._learnings_cache = combined
        return combined

    def load_heartbeat_rules(self) -> str:
        """Load HEARTBEAT.md content for autonomous operation."""
        return self._read_file(self.HEARTBEAT_FILE) or ""

    # ── Chat transcript ───────────────────────────────────────

    _MAX_TRANSCRIPT_SIZE = 2_000_000  # 2 MB — rotate if larger
    _GREETING_SENTINEL = ".greeting_seeded"

    def seed_bootstrap_greeting(self, greeting: str) -> bool:
        """Seed a one-shot greeting into the chat transcript (idempotent).

        Writes a single ``role=assistant`` entry tagged with
        ``_origin == "bootstrap_greeting"`` so the LLM context layer
        can distinguish it from a real model-authored message and
        skip provenance gates that would otherwise reject it.

        Idempotency is enforced via a sentinel file under the
        workspace root — once written, subsequent calls are no-ops
        even after a chat reset (which archives the transcript but
        leaves the sentinel intact). Returns ``True`` when the
        greeting was actually written, ``False`` when skipped.
        """
        if not greeting or not greeting.strip():
            return False
        sentinel = self.root / self._GREETING_SENTINEL
        if sentinel.exists():
            return False
        path = self.root / self.CHAT_TRANSCRIPT
        entry: dict = {
            "role": "assistant",
            "content": greeting,
            "ts": time.time(),
            "_origin": "bootstrap_greeting",
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(dumps_safe(entry) + "\n")
            sentinel.touch()
            return True
        except OSError as e:
            logger.debug("Failed to seed bootstrap greeting: %s", e)
            return False

    def append_chat_message(
        self, role: str, content: str, *,
        tool_names: list[str] | None = None,
        tools: list[dict] | None = None,
        turn_id: str | None = None,
        partial: bool = False,
    ) -> None:
        """Append a message to the persistent chat transcript (JSONL).

        Rotates (drops oldest half) when the file exceeds _MAX_TRANSCRIPT_SIZE
        to prevent unbounded growth in long-running sessions.

        ``turn_id`` + ``partial`` cooperate with :meth:`load_chat_transcript`
        to make in-flight assistant turns survive a dashboard refresh —
        when a turn fires a slow tool call (``await_task_event``, browser
        action) the loop writes a ``partial=True`` entry BEFORE tool
        dispatch, then overwrites it with a final ``partial=False`` entry
        when the turn closes. ``load_chat_transcript`` dedupes by
        ``turn_id`` and keeps the latest occurrence — legacy entries
        without ``turn_id`` are passed through unchanged.
        """
        path = self.root / self.CHAT_TRANSCRIPT
        entry: dict = {"role": role, "content": content, "ts": time.time()}
        if tools:
            entry["tools"] = tools
        elif tool_names:
            entry["tools"] = tool_names
        if turn_id:
            entry["turn_id"] = turn_id
        if partial:
            entry["partial"] = True
        try:
            with path.open("a") as f:
                f.write(dumps_safe(entry) + "\n")
            # Rotate if too large — keep newest half
            if path.stat().st_size > self._MAX_TRANSCRIPT_SIZE:
                lines = path.read_text(errors="replace").strip().split("\n")
                half = len(lines) // 2
                path.write_text("\n".join(lines[half:]) + "\n")
        except Exception as e:
            logger.debug("Failed to write chat transcript: %s", e)

    def load_chat_transcript(self, limit: int = 200) -> list[dict]:
        """Load recent messages from the persistent chat transcript.

        Dedupes by ``turn_id``: entries that share a ``turn_id`` collapse
        to the LATEST occurrence (the file is append-only, so later lines
        supersede earlier ones — i.e. the final assistant entry replaces
        the in-flight ``partial`` entry written before tool dispatch).
        Entries without a ``turn_id`` are passed through unchanged for
        back-compat with pre-fix transcripts. ``limit`` applies AFTER
        dedup so trimming can't drop a final entry that supersedes an
        earlier partial.
        """
        path = self.root / self.CHAT_TRANSCRIPT
        if not path.exists():
            return []
        try:
            text = path.read_text(errors="replace").strip()
            if not text:
                return []
            lines = text.split("\n")
            output: list[dict] = []
            turn_id_idx: dict[str, int] = {}
            for line in lines:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = entry.get("turn_id")
                if tid and tid in turn_id_idx:
                    # Later write wins — preserves chronological order of
                    # the FIRST occurrence so partial entries don't jump
                    # around the transcript when their final lands.
                    output[turn_id_idx[tid]] = entry
                else:
                    if tid:
                        turn_id_idx[tid] = len(output)
                    output.append(entry)
            return output[-limit:]
        except Exception as e:
            logger.debug("Failed to read chat transcript: %s", e)
            return []

    def archive_chat_transcript(self) -> None:
        """Archive the current transcript on chat reset."""
        path = self.root / self.CHAT_TRANSCRIPT
        if not path.exists():
            return
        try:
            if path.stat().st_size == 0:
                path.unlink(missing_ok=True)
                return
        except OSError:
            return
        archive_dir = self.root / self.CHAT_ARCHIVE_DIR
        archive_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        try:
            path.rename(archive_dir / f"{ts}.jsonl")
        except Exception as e:
            logger.debug("Failed to archive chat transcript: %s", e)

    # ── Activity log ────────────────────────────────────────

    def append_activity(
        self,
        trigger: str,
        summary: str,
        *,
        tools_used: list[str] | None = None,
        duration_ms: int = 0,
        tokens_used: int = 0,
        outcome: str = "ok",
        notifications: list[str] | None = None,
    ) -> None:
        """Append an entry to the activity log (JSONL).

        Used by heartbeat executions to record autonomous work separate from
        the chat transcript.  Rotates like chat_transcript when too large.
        """
        path = self.root / self.ACTIVITY_LOG
        entry: dict = {
            "trigger": trigger,
            "summary": summary,
            "ts": time.time(),
            "tools": tools_used or [],
            "duration_ms": duration_ms,
            "tokens_used": tokens_used,
            "outcome": outcome,
        }
        if notifications:
            entry["notifications"] = notifications
        try:
            with path.open("a") as f:
                f.write(dumps_safe(entry) + "\n")
            if path.stat().st_size > self._MAX_TRANSCRIPT_SIZE:
                lines = path.read_text(errors="replace").strip().split("\n")
                half = len(lines) // 2
                path.write_text("\n".join(lines[half:]) + "\n")
        except Exception as e:
            logger.debug("Failed to write activity log: %s", e)

    def load_activity(self, limit: int = 100) -> list[dict]:
        """Load recent entries from the activity log."""
        path = self.root / self.ACTIVITY_LOG
        if not path.exists():
            return []
        try:
            text = path.read_text(errors="replace").strip()
            if not text:
                return []
            lines = text.split("\n")
            entries = []
            for line in lines[-limit:]:
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return entries
        except Exception as e:
            logger.debug("Failed to read activity log: %s", e)
            return []

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

**Blackboard** — A shared key-value store for agent-to-agent data. Keys
are hierarchical (e.g. `status/researcher`, `research/acme_findings`).
Write data here that teammates need to read later. Always write your
current progress to `status/{your_id}` so teammates know what you're doing.
Use your domain as the key prefix for work output (e.g. `research/`,
`drafts/`, `analysis/`, `tasks/`). You need read/write permissions per
key pattern.

**Pub/Sub** — Ephemeral event topics for one-time signals. Publish
`research_complete` or `build_failed` to notify subscribed agents
instantly. Events are NOT stored — use the blackboard when data should
persist.

**When to use which:** Write to blackboard for data another agent needs
to read (now or later). Publish an event for "something just happened"
signals that trigger immediate reactions.

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
your budget with the `get_system_status` tool. Expensive operations: long
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

## Coordination Protocol

Use these tools to coordinate with teammates:

**Discovering teammates:**
→ `list_agents()` — roster with name, role, capabilities
→ `get_agent_profile(agent_id)` — read their INTERFACE.md contract
  plus live metadata (status, subscriptions, watches, recent writes)
Check a teammate's profile before your first coordination with them.

**Handing off work:**
→ `hand_off(to="agent_id", summary="what to do next", data="optional JSON")`
Writes your output to the blackboard, creates a task in their inbox, and
wakes them up. This is a single call — no separate publish needed.

**Checking for work:**
→ `check_inbox()`
Returns pending tasks from teammates with summaries and pointers to their
output. Call this on startup, during heartbeats, and when notified.
After processing a task, call `complete_task(task_key)` to mark it done.

**Sharing your state:**
→ `update_status(state="working|idle|blocked|done", summary="...")`
Teammates read your status to decide whether to wait or proceed.

**Your collaboration interface (INTERFACE.md):**
Describe what you accept (inputs, blackboard keys you read, events you
subscribe to), what you produce (outputs, keys you write, events you
publish), and how to send you work or feedback. Update via
`update_workspace`. Teammates read it via `get_agent_profile`.

**Reactive notifications (no polling needed):**
- `watch_blackboard(pattern)` — notified when matching keys change
- `subscribe_event(topic)` — notified on ephemeral one-time signals
Set these up once during setup — they persist across sessions.

**Three standard blackboard sections:**
- `status/{agent_id}` — each agent's current state
- `output/{agent_id}/{name}` — completed work products
- `tasks/{agent_id}/{task_id}` — pending work inbox

You can still use the lower-level tools (read_blackboard, write_blackboard,
publish_event) for custom data patterns (e.g. research/, drafts/, analysis/),
but prefer the coordination tools above for inter-agent workflows.

## Custom Skills

You can create reusable tools with `create_skill`. Custom skills are Python
functions decorated with `@skill` that get framework dependencies injected
automatically. Key capability: `mesh_client.browser_command(action, params)`
lets you build multi-step browser automation workflows — it's the same API
that browser_navigate, browser_click, and all browser tools use internally.
Call `reload_skills` after creating a skill to activate it.
"""


_SYSTEM_MD_PREAMBLE_STANDALONE = """\
# System Architecture

You are an agent running inside an isolated container in the OpenLegion mesh.
This document describes how the system works so you can operate effectively.

## How Your World Works

**Mesh host** — A central coordinator at your MESH_URL. Every external action
you take goes through it: LLM calls, web requests, notifications. You have
no direct internet access.

**Credential vault** — You never hold API keys. When you call an LLM or
external API, the mesh injects credentials server-side. You cannot leak
what you cannot see.

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
your budget with the `get_system_status` tool. Expensive operations: long
conversations (large context), vision/screenshot tools, embedding calls.

**Common errors and what they mean:**
- 429 Too Many Requests = rate limit hit (back off and retry)
- Budget exceeded = daily/monthly cap reached (wait or ask user to adjust)
- Tool loop detected = you called the same tool with same args too many times
"""


def generate_system_md(
    introspect_data: dict, agent_id: str, *, is_standalone: bool = False,
) -> str:
    """Generate SYSTEM.md content: static preamble only.

    The preamble teaches agents *how the system works*. Permissions and
    fleet data are provided live via the ``## Runtime Context`` block
    injected into the system prompt on each turn by ``AgentLoop``.

    Standalone agents get a trimmed preamble without blackboard, pub/sub,
    or multi-agent coordination sections.
    """
    content = _SYSTEM_MD_PREAMBLE_STANDALONE if is_standalone else _SYSTEM_MD_PREAMBLE
    if len(content) > _MAX_SYSTEM:
        content = content[:_MAX_SYSTEM].rsplit("\n", 1)[0] + "\n\n... (truncated)"
    return content


# ── BM25 implementation (no external deps) ───────────────────

_STOP_WORDS = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "i", "in", "is", "it", "of", "on", "or", "that",
    "the", "to", "was", "with",
])


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
