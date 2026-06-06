"""SKILL.md procedural skill packs — loader and store.

Skills are *instructions*, not code. A ``SKILL.md`` is YAML frontmatter +
a markdown body that tells the agent how to do a job using the tools it
already has. Unlike Tools (executable ``@tool`` functions in ``tools.py``),
a Skill adds no code to the trusted runtime — installing one never widens
the agent's tool permissions; it only composes grants the agent already
holds. The per-agent ACL stays the single capability chokepoint.

Wire-compatible with the agentskills.io ``SKILL.md`` standard: only
``name`` and ``description`` are required. Every other frontmatter field
(including vendor ``metadata.*`` namespaces) is preserved verbatim in
``Skill.metadata`` and interpreted later at use-time — never rejected here.

Progressive disclosure is implemented by the ``skills_list`` / ``skill_view``
tools in ``builtins/skills_tool.py``:

    Level 0  skills_list()           → names + descriptions (cheap catalog)
    Level 1  skill_view(name)        → full markdown body
    Level 2  skill_view(name, path)  → a bundled reference file
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.shared.utils import setup_logging

logger = setup_logging("agent.skills")

# Container-side store locations, mirroring the Tools layering:
#   bundled   → shipped read-only with the product (SKILLS_DIR, default /app/skills)
#   installed → operator/marketplace-installed packs (writable /data volume)
# ``installed`` overrides ``bundled`` on a name collision.
_BUNDLED_DIR_ENV = "SKILLS_DIR"
_DEFAULT_BUNDLED_DIR = "/app/skills"
_INSTALLED_DIR = "/data/skills"

# Skill names are matched by exact lookup, never used to build a path — but
# we still reject path-significant characters so an installed pack can't smuggle
# a traversal sequence into a name that later code might join onto a path.
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# Frontmatter fences are whole ``---`` lines (optional trailing whitespace; ``\r``
# tolerated for CRLF checkouts), so a ``---`` substring inside a YAML scalar can't
# be mistaken for a fence. One pattern serves both the opening-fence guard
# (``.match`` at pos 0) and the body split.
_FRONTMATTER_FENCE_RE = re.compile(r"^---[ \t\r]*$", re.MULTILINE)


@dataclass(frozen=True)
class Skill:
    """A parsed ``SKILL.md`` pack."""

    name: str
    description: str
    body: str
    directory: Path
    version: str = ""
    metadata: dict = field(default_factory=dict)
    source: str = ""  # "bundled" | "installed"


def _split_frontmatter(text: str) -> tuple[str, str] | None:
    """Split a SKILL.md into ``(yaml_frontmatter, body)``; None if absent.

    The ``---`` fences are matched as whole lines only (``^---$``), so a literal
    ``---`` inside a YAML scalar (e.g. a description containing ``---``) does not
    truncate the frontmatter. maxsplit=2 consumes just the opening and closing
    fences; any further ``---`` in the body (a horizontal rule) is left intact.
    """
    if not _FRONTMATTER_FENCE_RE.match(text):
        return None
    # ["", frontmatter, body]; the body keeps any further "---".
    parts = _FRONTMATTER_FENCE_RE.split(text, maxsplit=2)
    if len(parts) < 3:
        return None
    return parts[1], parts[2]


def parse_skill_md(md_path: Path, *, source: str = "") -> Skill | None:
    """Parse a single ``SKILL.md`` file. Returns None on any structural problem.

    Required frontmatter: ``name``, ``description``. Everything else is
    optional and preserved verbatim in ``Skill.metadata`` (minus the fields
    promoted to first-class attributes).
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.debug("Cannot read %s: %s", md_path, e)
        return None

    split = _split_frontmatter(text)
    if split is None:
        logger.debug("No YAML frontmatter in %s", md_path)
        return None
    raw_frontmatter, body = split

    try:
        import yaml
    except ImportError:
        logger.error("PyYAML is required to parse SKILL.md packs")
        return None
    try:
        meta = yaml.safe_load(raw_frontmatter)
    except Exception as e:
        logger.debug("Bad YAML frontmatter in %s: %s", md_path, e)
        return None
    if not isinstance(meta, dict):
        return None

    name = meta.get("name")
    description = meta.get("description")
    if not isinstance(name, str) or not isinstance(description, str):
        logger.debug("Skill %s missing name/description", md_path)
        return None
    name = name.strip()
    description = description.strip()
    if not name or not description:
        return None
    if not _NAME_RE.match(name):
        logger.warning("Skipping skill with unsafe name %r in %s", name, md_path)
        return None

    version = str(meta.get("version", "")).strip()
    # Preserve everything else verbatim (vendor namespaces, config, env, …).
    extra = {k: v for k, v in meta.items() if k not in {"name", "description", "version"}}

    return Skill(
        name=name,
        description=description,
        body=body.strip(),
        directory=md_path.parent,
        version=version,
        metadata=extra,
        source=source,
    )


class SkillStore:
    """Scans the bundled + installed skill directories on demand.

    The store is intentionally stateless — ``list``/``get`` re-scan disk on
    every call. Catalogs are tiny and scans are cheap, and statelessness
    means a freshly installed pack is visible immediately with no reload
    step (contrast Tools, which need an explicit ``reload_tools``).
    """

    def __init__(
        self,
        bundled_dir: str | os.PathLike | None = None,
        installed_dir: str | os.PathLike | None = None,
    ):
        self.bundled_dir = (
            Path(bundled_dir)
            if bundled_dir is not None
            else Path(os.environ.get(_BUNDLED_DIR_ENV, _DEFAULT_BUNDLED_DIR))
        )
        self.installed_dir = (
            Path(installed_dir) if installed_dir is not None else Path(_INSTALLED_DIR)
        )

    def _scan_dir(self, directory: Path, source: str) -> dict[str, Skill]:
        found: dict[str, Skill] = {}
        if not directory.is_dir():
            return found
        for md_path in sorted(directory.glob("**/SKILL.md")):
            skill = parse_skill_md(md_path, source=source)
            if skill is None:
                continue
            if skill.name in found:
                logger.warning(
                    "Duplicate skill %r in %s — keeping first", skill.name, source,
                )
                continue
            found[skill.name] = skill
        return found

    def _all(self) -> dict[str, Skill]:
        skills = self._scan_dir(self.bundled_dir, "bundled")
        # Installed packs override bundled on a name collision.
        skills.update(self._scan_dir(self.installed_dir, "installed"))
        return skills

    def list(self) -> list[Skill]:
        return sorted(self._all().values(), key=lambda s: s.name)

    def get(self, name: str) -> Skill | None:
        return self._all().get(name)

    def read_reference(self, name: str, rel_path: str) -> str | None:
        """Read a bundled reference file inside a skill (Level 2), traversal-safe.

        Resolves both the skill directory and the requested path, then verifies
        the target stays inside the skill — this catches ``..`` traversal and
        symlink escapes alike (a symlink pointing outside resolves outside the
        base and fails the containment check).
        """
        skill = self.get(name)
        if skill is None:
            return None
        base = skill.directory.resolve()
        target = (base / rel_path).resolve()
        if not target.is_relative_to(base):
            logger.warning("Path traversal blocked: %r in skill %r", rel_path, name)
            return None
        if not target.is_file():
            return None
        try:
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
