"""Tests for skill marketplace: install, list, remove."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from src.marketplace import (
    _parse_skill_manifest,
    _validate_all_skills,
    install_skill,
    list_skills,
    remove_skill,
)


def _create_valid_skill_dir(path: Path, name: str = "test-skill", version: str = "1.0.0") -> None:
    """Create a minimal valid marketplace skill directory."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\nversion: {version}\ndescription: A test skill\n---\n# {name}\n"
    )
    (path / "my_tool.py").write_text(
        'from src.agent.skills import skill\n\n'
        '@skill(name="my_tool", description="Test", parameters={"x": {"type": "string"}})\n'
        'def my_tool(x: str) -> dict:\n'
        '    return {"result": x}\n'
    )


# ── manifest parsing ─────────────────────────────────────────


class TestParseManifest:
    def test_parse_manifest_valid(self, tmp_path):
        _create_valid_skill_dir(tmp_path)
        meta = _parse_skill_manifest(tmp_path)
        assert meta is not None
        assert meta["name"] == "test-skill"
        assert meta["version"] == "1.0.0"
        assert meta["description"] == "A test skill"

    def test_parse_manifest_missing_fields(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "SKILL.md").write_text("---\nname: foo\n---\nBody\n")
        meta = _parse_skill_manifest(tmp_path)
        assert meta is None  # Missing version and description

    def test_parse_manifest_no_file(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        meta = _parse_skill_manifest(tmp_path)
        assert meta is None

    def test_parse_manifest_no_front_matter(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "SKILL.md").write_text("# Just a readme\nNo front matter.\n")
        meta = _parse_skill_manifest(tmp_path)
        assert meta is None


# ── install ───────────────────────────────────────────────────


class TestInstallSkill:
    def test_install_skill_success(self, tmp_path):
        """Mock git clone, valid skill + manifest."""
        marketplace_dir = tmp_path / "marketplace"

        def mock_clone(cmd, **kwargs):
            # Simulate git clone by creating the directory with valid content
            clone_target = cmd[-1]
            _create_valid_skill_dir(Path(clone_target))
            return type("Result", (), {"returncode": 0, "stderr": ""})()

        with patch("src.marketplace.subprocess.run", side_effect=mock_clone):
            result = install_skill("https://github.com/user/test-skill.git", marketplace_dir)

        assert result.get("installed") is True
        assert result["name"] == "test-skill"
        assert (marketplace_dir / "test-skill" / ".installed.json").exists()

    def test_install_skill_validation_failure(self, tmp_path):
        """Skill with eval() is rejected."""
        marketplace_dir = tmp_path / "marketplace"

        def mock_clone(cmd, **kwargs):
            clone_target = cmd[-1]
            target = Path(clone_target)
            target.mkdir(parents=True, exist_ok=True)
            (target / "SKILL.md").write_text(
                "---\nname: evil\nversion: 1.0\ndescription: Bad skill\n---\n"
            )
            (target / "evil.py").write_text(
                'from src.agent.skills import skill\n\n'
                '@skill(name="evil", description="Bad", parameters={})\n'
                'def evil() -> dict:\n'
                '    return eval("1+1")\n'
            )
            return type("Result", (), {"returncode": 0, "stderr": ""})()

        with patch("src.marketplace.subprocess.run", side_effect=mock_clone):
            result = install_skill("https://github.com/user/evil.git", marketplace_dir)

        assert "error" in result
        assert "validation" in result["error"].lower()

    def test_install_skill_no_manifest(self, tmp_path):
        """Error when no SKILL.md."""
        marketplace_dir = tmp_path / "marketplace"

        def mock_clone(cmd, **kwargs):
            clone_target = cmd[-1]
            target = Path(clone_target)
            target.mkdir(parents=True, exist_ok=True)
            (target / "tool.py").write_text("print('hello')\n")
            return type("Result", (), {"returncode": 0, "stderr": ""})()

        with patch("src.marketplace.subprocess.run", side_effect=mock_clone):
            result = install_skill("https://github.com/user/no-manifest.git", marketplace_dir)

        assert "error" in result
        assert "SKILL.md" in result["error"]

    def test_install_skill_git_failure(self, tmp_path):
        """Error when git clone fails."""
        marketplace_dir = tmp_path / "marketplace"

        def mock_clone(cmd, **kwargs):
            return type("Result", (), {"returncode": 128, "stderr": "repo not found"})()

        with patch("src.marketplace.subprocess.run", side_effect=mock_clone):
            result = install_skill("https://github.com/user/nonexistent.git", marketplace_dir)

        assert "error" in result
        assert "clone failed" in result["error"].lower()

    def test_install_skill_with_ref(self, tmp_path):
        """--ref flag is passed to git clone."""
        marketplace_dir = tmp_path / "marketplace"
        captured_cmds: list = []

        def mock_clone(cmd, **kwargs):
            captured_cmds.append(cmd)
            clone_target = cmd[-1]
            _create_valid_skill_dir(Path(clone_target))
            return type("Result", (), {"returncode": 0, "stderr": ""})()

        with patch("src.marketplace.subprocess.run", side_effect=mock_clone):
            install_skill("https://github.com/user/test-skill.git", marketplace_dir, ref="v2.0")

        assert "--branch" in captured_cmds[0]
        assert "v2.0" in captured_cmds[0]


# ── list / remove ─────────────────────────────────────────────


class TestListSkills:
    def test_list_skills(self, tmp_path):
        marketplace_dir = tmp_path / "marketplace"
        skill_dir = marketplace_dir / "my-skill"
        _create_valid_skill_dir(skill_dir)
        (skill_dir / ".installed.json").write_text(
            json.dumps({"name": "my-skill", "version": "1.0.0", "description": "Test"})
        )

        skills = list_skills(marketplace_dir)
        assert len(skills) == 1
        assert skills[0]["name"] == "my-skill"

    def test_list_skills_empty(self, tmp_path):
        skills = list_skills(tmp_path / "nonexistent")
        assert skills == []

    def test_list_skills_missing_metadata(self, tmp_path):
        """Skill dir without .installed.json still shows up."""
        marketplace_dir = tmp_path / "marketplace"
        (marketplace_dir / "orphan-skill").mkdir(parents=True)
        skills = list_skills(marketplace_dir)
        assert len(skills) == 1
        assert skills[0]["name"] == "orphan-skill"


class TestRemoveSkill:
    def test_remove_skill(self, tmp_path):
        marketplace_dir = tmp_path / "marketplace"
        skill_dir = marketplace_dir / "my-skill"
        _create_valid_skill_dir(skill_dir)

        result = remove_skill("my-skill", marketplace_dir)
        assert result["removed"] is True
        assert not skill_dir.exists()

    def test_remove_nonexistent(self, tmp_path):
        marketplace_dir = tmp_path / "marketplace"
        marketplace_dir.mkdir(parents=True)

        result = remove_skill("ghost", marketplace_dir)
        assert "error" in result
        assert "not found" in result["error"].lower()


# ── code validation ───────────────────────────────────────────


class TestValidateAllSkills:
    def test_validates_clean_code(self, tmp_path):
        _create_valid_skill_dir(tmp_path)
        errors = _validate_all_skills(tmp_path)
        assert errors == []

    def test_catches_forbidden_call(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "bad.py").write_text(
            'from src.agent.skills import skill\n\n'
            '@skill(name="bad", description="Bad", parameters={})\n'
            'def bad() -> dict:\n'
            '    return eval("1+1")\n'
        )
        errors = _validate_all_skills(tmp_path)
        assert len(errors) == 1
        assert "eval" in errors[0]
