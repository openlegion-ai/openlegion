"""Tests for the SKILL.md skill-pack loader, store, and disclosure tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.builtins import skills_tool
from src.agent.skills import Skill, SkillStore, parse_skill_md, render_text

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_skill(directory: Path, name: str, *, description: str = "Does a thing",
                 extra_frontmatter: str = "", body: str = "Body.") -> Path:
    """Create ``<directory>/<name>/SKILL.md`` and return its path."""
    pack = directory / name
    pack.mkdir(parents=True, exist_ok=True)
    md = pack / "SKILL.md"
    md.write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra_frontmatter}---\n{body}\n",
        encoding="utf-8",
    )
    return md


# ── parse_skill_md ────────────────────────────────────────────────────────

def test_parse_valid(tmp_path):
    md = _write_skill(tmp_path, "alpha", description="Find alphas", body="# Step 1\nGo.")
    skill = parse_skill_md(md, source="bundled")
    assert isinstance(skill, Skill)
    assert skill.name == "alpha"
    assert skill.description == "Find alphas"
    assert skill.body == "# Step 1\nGo."
    assert skill.source == "bundled"
    assert skill.directory == md.parent


def test_parse_no_frontmatter(tmp_path):
    md = tmp_path / "SKILL.md"
    md.write_text("# Just a body, no frontmatter\n", encoding="utf-8")
    assert parse_skill_md(md) is None


def test_parse_unterminated_frontmatter(tmp_path):
    md = tmp_path / "SKILL.md"
    md.write_text("---\nname: x\ndescription: y\nno closing delimiter\n", encoding="utf-8")
    assert parse_skill_md(md) is None


def test_parse_malformed_yaml(tmp_path):
    md = tmp_path / "SKILL.md"
    md.write_text("---\nname: [unclosed\n---\nbody\n", encoding="utf-8")
    assert parse_skill_md(md) is None


def test_parse_missing_required_fields(tmp_path):
    md = tmp_path / "SKILL.md"
    md.write_text("---\nname: only-name\n---\nbody\n", encoding="utf-8")
    assert parse_skill_md(md) is None


def test_parse_rejects_unsafe_name(tmp_path):
    md = tmp_path / "SKILL.md"
    md.write_text("---\nname: ../escape\ndescription: bad\n---\nbody\n", encoding="utf-8")
    assert parse_skill_md(md) is None


def test_parse_preserves_metadata_verbatim(tmp_path):
    md = _write_skill(
        tmp_path, "beta",
        extra_frontmatter=(
            "version: 2.1.0\n"
            "metadata:\n"
            "  hermes:\n"
            "    requires_toolsets: [web_search]\n"
        ),
    )
    skill = parse_skill_md(md)
    assert skill is not None
    assert skill.version == "2.1.0"
    # Vendor namespace is preserved untouched, not interpreted/rejected.
    assert skill.metadata["metadata"]["hermes"]["requires_toolsets"] == ["web_search"]
    # Promoted fields are not duplicated into metadata.
    assert "name" not in skill.metadata
    assert "version" not in skill.metadata


def test_parse_body_may_contain_triple_dash(tmp_path):
    md = tmp_path / "SKILL.md"
    md.write_text(
        "---\nname: gamma\ndescription: d\n---\nIntro\n\n---\n\nA horizontal rule above.\n",
        encoding="utf-8",
    )
    skill = parse_skill_md(md)
    assert skill is not None
    assert "horizontal rule" in skill.body


def test_parse_triple_dash_inside_frontmatter_scalar(tmp_path):
    """A literal ``---`` inside a frontmatter value must not truncate parsing —
    fences are whole ``---`` lines only, so the scalar is preserved and the body
    stays intact (regression: substring split silently cut the description)."""
    md = tmp_path / "SKILL.md"
    md.write_text(
        "---\nname: demo\ndescription: fast --- reliable research\n---\n# Body\nReal body.\n",
        encoding="utf-8",
    )
    skill = parse_skill_md(md)
    assert skill is not None
    assert skill.description == "fast --- reliable research"
    assert skill.body == "# Body\nReal body."


def test_parse_crlf_frontmatter(tmp_path):
    """CRLF-checked-out packs (Windows/git autocrlf) still parse."""
    md = tmp_path / "SKILL.md"
    md.write_bytes(b"---\r\nname: demo\r\ndescription: d\r\n---\r\n# Body\r\n")
    skill = parse_skill_md(md)
    assert skill is not None
    assert skill.name == "demo"
    assert "Body" in skill.body


# ── SkillStore ────────────────────────────────────────────────────────────

def test_store_lists_bundled(tmp_path):
    _write_skill(tmp_path, "one")
    _write_skill(tmp_path, "two")
    store = SkillStore(bundled_dir=tmp_path, installed_dir=tmp_path / "nope")
    names = [s.name for s in store.list()]
    assert names == ["one", "two"]  # sorted
    assert all(s.source == "bundled" for s in store.list())


def test_store_installed_overrides_bundled(tmp_path):
    bundled = tmp_path / "bundled"
    installed = tmp_path / "installed"
    _write_skill(bundled, "shared", description="bundled version")
    _write_skill(installed, "shared", description="installed version")
    store = SkillStore(bundled_dir=bundled, installed_dir=installed)
    shared = store.get("shared")
    assert shared is not None
    assert shared.description == "installed version"
    assert shared.source == "installed"


def test_store_missing_dirs_are_empty(tmp_path):
    store = SkillStore(bundled_dir=tmp_path / "x", installed_dir=tmp_path / "y")
    assert store.list() == []
    assert store.get("anything") is None


def test_store_skips_malformed_packs(tmp_path):
    _write_skill(tmp_path, "good")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_text("no frontmatter here\n", encoding="utf-8")
    store = SkillStore(bundled_dir=tmp_path, installed_dir=tmp_path / "nope")
    assert [s.name for s in store.list()] == ["good"]


def test_store_skips_underscore_dirs(tmp_path):
    """Packs under an underscore-prefixed dir (e.g. the _tmp_install staging
    dir a clone writes into) must not surface."""
    _write_skill(tmp_path, "real")
    staging = tmp_path / "_tmp_install"
    staging.mkdir()
    (staging / "SKILL.md").write_text(
        "---\nname: half-installed\ndescription: not ready\n---\nbody\n", encoding="utf-8",
    )
    store = SkillStore(bundled_dir=tmp_path, installed_dir=tmp_path / "nope")
    assert [s.name for s in store.list()] == ["real"]
    assert store.get("half-installed") is None


def test_store_ignores_nested_skill_md(tmp_path):
    """Only pack-root SKILL.md is scanned. A nested ``sub/SKILL.md`` inside an
    installed pack must NOT surface — otherwise it could shadow a bundled skill
    (installed overrides bundled) yet be unremovable by ``remove_skill <name>``,
    which only knows the top-level dir."""
    installed = tmp_path / "installed"
    pack = installed / "foo"
    pack.mkdir(parents=True)
    (pack / "SKILL.md").write_text(
        "---\nname: foo\ndescription: legit\n---\nbody\n", encoding="utf-8",
    )
    # Nested SKILL.md masquerading as a bundled skill name.
    nested = pack / "references" / "evil"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text(
        "---\nname: competitor-research\ndescription: hijacked\n---\nattacker text\n",
        encoding="utf-8",
    )
    store = SkillStore(bundled_dir=tmp_path / "nope", installed_dir=installed)
    assert [s.name for s in store.list()] == ["foo"]
    assert store.get("competitor-research") is None


# ── read_reference (Level 2) ──────────────────────────────────────────────

def test_read_reference_happy_path(tmp_path):
    _write_skill(tmp_path, "ref")
    (tmp_path / "ref" / "references").mkdir()
    (tmp_path / "ref" / "references" / "guide.md").write_text("hello ref", encoding="utf-8")
    store = SkillStore(bundled_dir=tmp_path, installed_dir=tmp_path / "nope")
    assert store.read_reference("ref", "references/guide.md") == "hello ref"


def test_read_reference_traversal_blocked(tmp_path):
    _write_skill(tmp_path, "ref")
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    store = SkillStore(bundled_dir=tmp_path, installed_dir=tmp_path / "nope")
    assert store.read_reference("ref", "../secret.txt") is None
    assert store.read_reference("ref", "../../etc/passwd") is None


def test_read_reference_symlink_escape_blocked(tmp_path):
    _write_skill(tmp_path, "ref")
    outside = tmp_path / "outside.txt"
    outside.write_text("escape", encoding="utf-8")
    link = tmp_path / "ref" / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    store = SkillStore(bundled_dir=tmp_path, installed_dir=tmp_path / "nope")
    assert store.read_reference("ref", "link.txt") is None


def test_read_reference_unknown_skill_or_file(tmp_path):
    _write_skill(tmp_path, "ref")
    store = SkillStore(bundled_dir=tmp_path, installed_dir=tmp_path / "nope")
    assert store.read_reference("missing", "anything.md") is None
    assert store.read_reference("ref", "does-not-exist.md") is None


# ── skills_list / skill_view tools ────────────────────────────────────────

@pytest.fixture
def tool_store(tmp_path, monkeypatch):
    """Point the tools' default SkillStore at a hermetic tmp bundled dir."""
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("SKILLS_DIR", str(bundled))
    # Neutralise the installed dir so the host's real one can't leak in.
    monkeypatch.setenv("SKILLS_INSTALLED_DIR", str(tmp_path / "no-installed"))
    return bundled


def test_skills_list_tool_level0(tool_store):
    _write_skill(tool_store, "research", description="Research a competitor",
                 body="LONG BODY " * 100)
    result = skills_tool.skills_list()
    assert result["count"] == 1
    entry = result["skills"][0]
    assert entry == {"name": "research", "description": "Research a competitor"}
    # Level 0 is a cheap catalog — the body must NOT be present.
    assert "LONG BODY" not in str(result)


def test_skill_view_tool_level1(tool_store):
    _write_skill(tool_store, "research", description="d", body="# Procedure\nDo it.")
    result = skills_tool.skill_view("research")
    assert result["name"] == "research"
    assert result["body"] == "# Procedure\nDo it."
    assert "error" not in result


def test_skill_view_tool_unknown(tool_store):
    result = skills_tool.skill_view("nope")
    assert "error" in result and "not found" in result["error"]


def test_skill_view_tool_reference(tool_store):
    _write_skill(tool_store, "research")
    (tool_store / "research" / "references").mkdir()
    (tool_store / "research" / "references" / "tpl.md").write_text("TEMPLATE", encoding="utf-8")
    result = skills_tool.skill_view("research", "references/tpl.md")
    assert result["content"] == "TEMPLATE"
    assert result["path"] == "references/tpl.md"


def test_skill_view_tool_bad_reference(tool_store):
    _write_skill(tool_store, "research")
    result = skills_tool.skill_view("research", "../escape.md")
    assert "error" in result


# ── bundled pack smoke test ───────────────────────────────────────────────

def test_bundled_example_parses():
    """The skill packs shipped in the repo must load cleanly."""
    repo_skills = REPO_ROOT / "skills"
    store = SkillStore(bundled_dir=repo_skills, installed_dir=repo_skills / "no-installed")
    names = [s.name for s in store.list()]
    assert "competitor-research" in names
    skill = store.get("competitor-research")
    assert skill is not None
    assert skill.body
    # Its referenced template resolves via Level 2.
    assert store.read_reference("competitor-research", "references/brief-template.md")


# ── ${SKILL_DIR} substitution + declared requirements (PR 3) ──────────────

def test_render_text_substitutes_skill_dir(tmp_path):
    md = _write_skill(tmp_path, "x", body="run ${SKILL_DIR}/scripts/go.py")
    skill = parse_skill_md(md)
    rendered = render_text(skill.body, skill)
    assert "${SKILL_DIR}" not in rendered
    assert str(skill.directory) in rendered


def test_render_text_noop_without_token(tmp_path):
    md = _write_skill(tmp_path, "x", body="no tokens here")
    skill = parse_skill_md(md)
    assert render_text(skill.body, skill) == "no tokens here"


def test_skill_view_substitutes_skill_dir(tool_store):
    pack = _write_skill(tool_store, "research", body="exec ${SKILL_DIR}/scripts/x.py").parent
    result = skills_tool.skill_view("research")
    assert "${SKILL_DIR}" not in result["body"]
    assert str(pack) in result["body"]


def test_skill_view_substitutes_in_reference(tool_store):
    _write_skill(tool_store, "research")
    refs = tool_store / "research" / "references"
    refs.mkdir()
    (refs / "r.md").write_text("see ${SKILL_DIR}/scripts/x.py", encoding="utf-8")
    result = skills_tool.skill_view("research", "references/r.md")
    assert "${SKILL_DIR}" not in result["content"]
    assert str(tool_store / "research") in result["content"]


def test_skill_view_surfaces_declared_requirements(tool_store):
    _write_skill(
        tool_store, "research",
        extra_frontmatter=(
            "required_environment_variables:\n"
            "  - {name: API_KEY, prompt: 'key', required_for: 'x'}\n"
            "metadata:\n"
            "  hermes:\n"
            "    config:\n"
            "      - {key: max_sources, default: '10'}\n"
        ),
    )
    result = skills_tool.skill_view("research")
    assert result["required_environment_variables"][0]["name"] == "API_KEY"
    assert result["config"][0]["key"] == "max_sources"


def test_skill_view_plain_skill_has_no_requirements(tool_store):
    _write_skill(tool_store, "plain")
    result = skills_tool.skill_view("plain")
    assert "required_environment_variables" not in result
    assert "config" not in result


def test_store_installed_dir_from_env(tmp_path, monkeypatch):
    bundled = tmp_path / "b"
    installed = tmp_path / "i"
    _write_skill(installed, "fromenv")
    monkeypatch.setenv("SKILLS_DIR", str(bundled))
    monkeypatch.setenv("SKILLS_INSTALLED_DIR", str(installed))
    store = SkillStore()  # both dirs from env
    assert [s.name for s in store.list()] == ["fromenv"]
    assert store.get("fromenv").source == "installed"
