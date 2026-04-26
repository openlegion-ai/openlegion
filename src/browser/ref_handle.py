"""Rich ref-identity types for the browser service (Phase 1.2).

Replaces the flat ``dict[str, dict]`` that previously backed ``inst.refs`` with
a dataclass carrying everything needed to resolve a snapshot ref back to a
concrete Playwright element across:

* **Tabs** — ``page_id`` identifies the Page the ref came from. Survives
  navigation within that tab; becomes stale when the tab closes.
* **Frames** — ``frame_id`` identifies the Frame. ``None`` means main frame.
  Per-frame UUIDs (not URL hashes) disambiguate sibling unnamed iframes with
  the same ``src``.
* **Shadow DOM** — ``shadow_path`` walks through open shadow roots. Each hop
  carries ``(selector, occurrence, discriminator)`` so sibling shadow hosts
  with identical selectors don't produce ambiguous resolution.
* **Modal scoping** — ``scope_root`` is the modal selector the snapshot was
  scoped to (or ``None``).

Downstream features built on this:

* Cross-frame diff mode (§7.3) — refs collide safely across frames because
  frame_id is part of identity.
* Iframe traversal (§8.4) — extends populated frame_id; no handle-shape change.
* Shadow DOM walker (§8.3) — populates ``shadow_path``; no handle-shape change.

The handle shape is finalized now so later phases don't force another
refactor.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# ── Shadow DOM path segments ────────────────────────────────────────────────


@dataclass(frozen=True)
class ShadowHop:
    """One hop from outer to inner shadow root during shadow-DOM descent.

    Stored ordered outermost-shadow → innermost in ``RefHandle.shadow_path``.

    Attributes:
        selector: CSS selector that matches the shadow host in its parent
            (light-DOM or outer shadow) context. Example: ``"my-card"`` or
            ``"div.panel > custom-widget"``.
        occurrence: Position among siblings that match ``selector``. Required
            because two sibling ``<my-card>`` hosts share the same selector.
        discriminator: Stable per-host identity that survives DOM reordering
            between snapshot and resolution. Priority:
                1. ``data-testid`` / ``data-test`` / ``data-qa`` attr value
                2. ``id`` attr if it looks stable (NOT UUID-shaped,
                   NOT React auto-id like ``"r:R123:"``)
                3. Content hash of host's ``outerHTML`` with children
                   stripped (shape-only identity)
            Resolution verifies ``discriminator`` matches the host found at
            ``selector[occurrence]``; mismatch raises ``RefStale`` rather
            than silently landing in the wrong shadow root.
    """

    selector: str
    occurrence: int
    discriminator: str

    def __post_init__(self) -> None:
        if not self.selector:
            raise ValueError("ShadowHop.selector must be non-empty")
        if not self.discriminator:
            raise ValueError("ShadowHop.discriminator must be non-empty")


# ── Ref handle (the canonical identity of a snapshot ref) ───────────────────


@dataclass(frozen=True)
class RefHandle:
    """Rich ref identity; replaces the former ``dict[str, dict]`` per-ref entry.

    ``RefHandle`` is strictly an INTERNAL structure. The agent-facing wire
    format (the ``refs`` field in ``browser_get_elements`` responses) remains
    a JSON dict of the minimal shape agents already expect. Use
    :meth:`to_agent_dict` to produce the wire shape.

    Attributes:
        page_id: Stable UUID assigned to a Playwright ``Page`` at creation.
            Stored in ``CamoufoxInstance.page_ids[page]``. Survives navigation
            within that tab. On tab close, refs pointing at this page_id are
            stale (raise :class:`RefStale` at resolution).
        frame_id: ``None`` for main frame. Else per-frame UUID assigned at
            first observation and stored in
            ``CamoufoxInstance.frame_ids: WeakKeyDictionary[Frame, str]``.
            URL alone is NOT unique (duplicate unnamed iframes at same ``src``
            would collide). UUIDs are stable for the life of the Frame
            object; on navigation within the tab, new frames get new UUIDs.
        shadow_path: Empty tuple = light DOM. Else ordered
            :class:`ShadowHop` sequence from outermost-shadow to innermost.
            Required to disambiguate identical-looking elements across
            distinct shadow roots.
        scope_root: When snapshot was scoped to a modal, the modal selector
            (e.g. ``'[role="dialog"]'``). ``None`` otherwise. Used to bound
            ``_locator_from_ref`` queries so occurrence indices match the
            scoped snapshot.
        role: ARIA role (explicit or derived from tag) at snapshot time.
        name: Accessible name at snapshot time.
        occurrence: Position among refs sharing the same
            ``(role, name, shadow_path, scope_root)`` — the disambiguator
            already present in v1. Folded into ``element_key`` too.
        disabled: Disabled state at snapshot time (mirrors existing dict
            field).
        element_key: Stability hash used for cross-snapshot ref identity
            (Phase 4.3 diff-mode). Priority chain:
                1. ``data-testid`` / ``data-test`` / ``data-qa`` attr value
                2. ``id`` attr when stable (not UUID-shaped / not auto-id)
                3. ``SHA1(role, accessible_name, nearest_landmark)``
                4. Structural fallback:
                   ``SHA1(role, accessible_name, parent_hash, sibling_index)``
            Folded with ``shadow_path`` + ``frame_id`` so identical light-DOM
            refs in different frames / shadow roots get different keys.
    """

    page_id: str
    frame_id: str | None
    shadow_path: tuple[ShadowHop, ...]
    scope_root: str | None
    role: str
    name: str
    occurrence: int
    disabled: bool
    element_key: str = ""   # populated by snapshot; empty-string default keeps
                            # bootstrapping simple before shadow/frame work

    # ── Wire-format conversion ──────────────────────────────────────────────

    def to_agent_dict(self) -> dict:
        """Return the minimal dict shape agents have seen since v1.

        Stable public surface — do not add fields here without updating
        ``browser_get_elements`` callers and the agent-tools documentation.
        """
        return {
            "role": self.role,
            "name": self.name,
            "index": self.occurrence,
            "disabled": self.disabled,
        }

    # ── Factory helpers ─────────────────────────────────────────────────────

    @classmethod
    def light_dom(
        cls,
        *,
        page_id: str,
        scope_root: str | None,
        role: str,
        name: str,
        occurrence: int,
        disabled: bool,
        element_key: str = "",
    ) -> "RefHandle":
        """Build a light-DOM, main-frame handle (the common case today).

        Keeps migration simple — existing snapshot sites need just
        ``page_id`` / ``scope_root`` to be RefHandle-correct without knowing
        about shadow or iframe fields yet.
        """
        return cls(
            page_id=page_id,
            frame_id=None,
            shadow_path=(),
            scope_root=scope_root,
            role=role,
            name=name,
            occurrence=occurrence,
            disabled=disabled,
            element_key=element_key,
        )

    @classmethod
    def shadow(
        cls,
        *,
        page_id: str,
        scope_root: str | None,
        shadow_path: tuple[ShadowHop, ...],
        role: str,
        name: str,
        occurrence: int,
        disabled: bool,
        element_key: str = "",
        frame_id: str | None = None,
    ) -> "RefHandle":
        """Build a handle that points inside one or more open shadow roots.

        ``shadow_path`` must be non-empty; the outermost shadow host is
        first. Empty ``shadow_path`` would be a light-DOM ref — use
        :meth:`light_dom` instead so the call site reads correctly.
        """
        if not shadow_path:
            raise ValueError("shadow_path must be non-empty for RefHandle.shadow")
        return cls(
            page_id=page_id,
            frame_id=frame_id,
            shadow_path=shadow_path,
            scope_root=scope_root,
            role=role,
            name=name,
            occurrence=occurrence,
            disabled=disabled,
            element_key=element_key,
        )


# ── Element-key computation ─────────────────────────────────────────────────


def compute_element_key(
    *,
    role: str,
    name: str,
    landmark: str = "",
    test_id: str | None = None,
    dom_id: str | None = None,
    parent_hash: str = "",
    sibling_index: int = 0,
    frame_id: str | None = None,
    shadow_path: tuple[ShadowHop, ...] = (),
) -> str:
    """Compute a stable element key from the strongest available signal.

    Priority chain (highest → lowest):

    1. ``data-testid`` / ``data-test`` / ``data-qa`` — author-intentional
       stable identity. Prefer.
    2. ``id`` attribute *if* stable-looking. Reject generated ids:
        * UUIDs — ``^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-...``
        * React 18 ``useId`` format — ``^[:r][0-9a-z:]+:?$``
        * Radix-style ``data-radix-*`` ids
    3. ``SHA1(role, accessible_name, landmark)`` — survives React reconcile
       as long as the accessible name + surrounding landmark are stable.
    4. Structural fallback: includes parent hash + sibling index.

    The ``frame_id`` and ``shadow_path`` are folded into the final hash so
    elements in distinct frames / shadow roots never collide even if all
    other signals match.
    """
    ctx = f"frame={frame_id}|shadow={_shadow_digest(shadow_path)}"

    # Priority 1: explicit test attributes
    if test_id:
        return _mix(ctx, f"testid:{test_id}")

    # Priority 2: stable DOM id
    if dom_id and _is_stable_id(dom_id):
        return _mix(ctx, f"id:{dom_id}")

    # Priority 3: role + accessible name + landmark
    if name:
        return _mix(ctx, f"rnl:{role}/{name}/{landmark}")

    # Priority 4: structural fallback
    return _mix(ctx, f"struct:{role}/{landmark}/{parent_hash}/{sibling_index}")


# ── Private helpers ─────────────────────────────────────────────────────────


def _mix(ctx: str, payload: str) -> str:
    """Short stable hash. Length-16 hex is ample for per-page uniqueness."""
    h = hashlib.sha1(f"{ctx}|{payload}".encode("utf-8"))
    return h.hexdigest()[:16]


def _shadow_digest(path: tuple[ShadowHop, ...]) -> str:
    """Deterministic summary of a shadow path for hashing context."""
    if not path:
        return ""
    parts = [f"{h.selector}#{h.occurrence}@{h.discriminator}" for h in path]
    return "/".join(parts)


# Heuristics for "does this id look stable?"
# Rejects: UUID v4, React 18 useId, long hex blobs. Accepts short
# human-author-looking ids.

_UUID_SHAPE = "00000000-0000-0000-0000-000000000000"


def _is_stable_id(candidate: str) -> bool:
    """Return True when ``candidate`` looks author-chosen, not generated."""
    if not candidate:
        return False
    # React useId shape:  ":r0:", ":R12aB:", "r:r0:"
    if candidate.startswith((":r", "r:", ":R")) and candidate.endswith(":"):
        return False
    # UUID v4 shape (eight-four-four-four-twelve hex)
    if (
        len(candidate) == len(_UUID_SHAPE)
        and candidate[8] == "-" and candidate[13] == "-"
        and candidate[18] == "-" and candidate[23] == "-"
        and all(c in "0123456789abcdef-" for c in candidate.lower())
    ):
        return False
    # Long pure-hex blobs (generated at build/runtime)
    if len(candidate) >= 16 and all(c in "0123456789abcdef" for c in candidate.lower()):
        return False
    return True


# ── Errors ──────────────────────────────────────────────────────────────────


def from_legacy_dict(
    d: dict,
    *,
    page_id: str = "test-page",
    scope_root: str | None = None,
) -> "RefHandle":
    """Build a RefHandle from the v1 minimal dict shape.

    Used by migrated test fixtures that previously seeded
    ``inst.refs = {"e0": {"role": ..., "name": ..., "index": ..., "disabled": ...}}``.
    Production code should construct :class:`RefHandle` directly via
    :meth:`RefHandle.light_dom` or the dataclass constructor.
    """
    return RefHandle.light_dom(
        page_id=page_id,
        scope_root=scope_root,
        role=d.get("role", ""),
        name=d.get("name", ""),
        occurrence=d.get("index", 0),
        disabled=bool(d.get("disabled", False)),
    )


class RefStale(Exception):
    """Raised when a ref points to a tab/frame that no longer exists.

    Distinct from ``not_found`` (ref shape valid, element genuinely absent).
    Lets callers distinguish "retry with fresh snapshot" (stale) from
    "element genuinely missing on current page" (not_found).

    Raising sites:
        * ``RefHandle.page_id`` not in ``inst.page_ids_inv`` → tab closed.
        * ``RefHandle.frame_id`` not live in ``inst.frame_ids_inv`` → frame
          detached.
        * Shadow-path resolution fails because a host's ``discriminator``
          no longer matches (§8.3).
    """

    def __init__(self, reason: str, *, ref: str | None = None):
        self.reason = reason
        self.ref = ref
        msg = f"Ref stale: {reason}"
        if ref:
            msg += f" (ref={ref})"
        super().__init__(msg)
