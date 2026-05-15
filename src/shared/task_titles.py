"""Title-length policy for task records.

Lives in ``src/shared`` because both the mesh-side task store
(``src.host.orchestration.Tasks.create``) and the agent-side
coordination tool (``src.agent.builtins.coordination_tool._hand_off_v2``)
apply this policy. Agent containers ship only ``src/agent`` +
``src/shared``, so any helper they reach for must live here — importing
from ``src.host`` across the trust boundary fails with
``ModuleNotFoundError`` inside the agent container.

``MAX_TITLE_CHARS`` is a hard cap. ``LONG_TITLE_THRESHOLD`` is the soft
ceiling: when a caller submits a single ``title`` longer than this and no
separate ``description``, the long string becomes the description and a
short label is derived for the title. Defends against agents that stuff
full instructions into the title field — the dashboard expects a
one-line label, not a paragraph.
"""

from __future__ import annotations

MAX_TITLE_CHARS: int = 200
LONG_TITLE_THRESHOLD: int = 100
SHORT_TITLE_TARGET: int = 80


def _derive_short_title(text: str) -> str:
    """Extract a short single-line label from a longer task instruction.

    Strategy: take the first non-empty line, then split on sentence
    terminators (``.``, ``!``, ``?``) and the en/em dash separators we
    see in handoff phrasings ("Draft Q3 — full brief..."). Pick the
    first chunk; if it's still too long, hard-cut at ``SHORT_TITLE_TARGET``
    with an ellipsis. Whitespace-collapsed so multi-line inputs don't
    leave dangling indents.

    Returns ``""`` for empty or whitespace-only input — callers (the
    title-normalizer / ``Tasks.create``) handle that fallback.
    """
    if not text or not text.strip():
        return ""
    # First non-empty line — handles "Subject\nBody" payloads cleanly.
    first_line = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped:
            first_line = stripped
            break
    if not first_line:
        first_line = text.strip()
    # Split on sentence + dash boundaries; pick the first chunk that
    # actually carries content (skip empty leading splits).
    candidate = first_line
    for sep in (". ", "! ", "? ", " — ", " - ", ": "):
        if sep in candidate:
            head = candidate.split(sep, 1)[0].strip()
            if head:
                candidate = head
                break
    # Collapse internal whitespace so a copy-pasted blob doesn't render
    # ragged in a one-line truncate.
    candidate = " ".join(candidate.split())
    if len(candidate) > SHORT_TITLE_TARGET:
        # Prefer cutting on a word boundary near the cap so we don't end
        # mid-word. ``rsplit`` returns the original string if no space.
        cut = candidate[:SHORT_TITLE_TARGET].rsplit(" ", 1)[0]
        if len(cut) < SHORT_TITLE_TARGET // 2:
            cut = candidate[:SHORT_TITLE_TARGET]
        candidate = cut.rstrip(",;:") + "…"
    return candidate


def _normalize_title_and_description(
    title: str, description: str | None,
) -> tuple[str, str | None]:
    """Apply the title-length policy.

    Three cases:

    * Title is short (≤ ``LONG_TITLE_THRESHOLD``): pass through unchanged.
    * Title is long but caller provided a separate ``description``: trust
      the caller (they made an intentional choice), just hard-cap title
      at ``MAX_TITLE_CHARS``.
    * Title is long and no description: split — the long string becomes
      the description (preserved verbatim) and we derive a short label
      for the title. This is the wall-of-text recovery path.
    """
    if not title:
        return title, description
    if len(title) <= LONG_TITLE_THRESHOLD:
        return title, description
    if description:
        # Caller already split — respect their choice, just cap title so
        # one bad input can't blow past the dashboard layout.
        return title[:MAX_TITLE_CHARS], description
    # Long title, no description: treat title as description, derive
    # a short title from it.
    short = _derive_short_title(title)
    if not short:
        # Pathological input (whitespace-only after splitting). Fall
        # back to a hard cut so we always have *some* title.
        short = title[:SHORT_TITLE_TARGET].rstrip() + "…"
    return short, title
