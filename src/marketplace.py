"""Git-based skill marketplace for installing community skills.

Skills are cloned from git repos, validated against the existing AST
checker, and mounted into agent containers at runtime.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from src.shared.utils import setup_logging

logger = setup_logging("marketplace")


def _parse_skill_manifest(path: Path) -> dict | None:
    """Parse YAML front matter from a SKILL.md manifest.

    Expected format::

        ---
        name: my-skill
        version: 1.0.0
        description: What this skill does
        ---
        # Optional markdown body...

    Returns dict with at least name, version, description, or None on failure.
    """
    skill_md = path / "SKILL.md"
    if not skill_md.exists():
        return None

    text = skill_md.read_text()
    if not text.startswith("---"):
        return None

    # Extract YAML front matter between --- delimiters
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None

    try:
        import yaml
        meta = yaml.safe_load(parts[1])
    except Exception:
        return None

    if not isinstance(meta, dict):
        return None

    # Require name, version, description
    for field in ("name", "version", "description"):
        if field not in meta:
            return None

    return meta


def _validate_all_skills(directory: Path) -> list[str]:
    """Run AST validation on all .py files in a directory.

    Returns list of error messages (empty = all valid).
    """
    from src.agent.builtins.skill_tool import _validate_skill_code

    errors: list[str] = []
    for py_file in directory.glob("**/*.py"):
        if py_file.name.startswith("_"):
            continue
        code = py_file.read_text()
        error = _validate_skill_code(code)
        if error:
            errors.append(f"{py_file.name}: {error}")
    return errors


def install_skill(
    repo_url: str,
    marketplace_dir: Path,
    ref: str = "",
) -> dict:
    """Clone a git repo, validate manifest + code, install to marketplace.

    Returns dict with status info or error.
    """
    marketplace_dir.mkdir(parents=True, exist_ok=True)

    # Clone to temp directory first
    tmp_dir = marketplace_dir / "_tmp_install"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    clone_cmd = ["git", "clone", "--depth", "1"]
    if ref:
        clone_cmd += ["--branch", ref]
    clone_cmd += [repo_url, str(tmp_dir)]

    result = subprocess.run(
        clone_cmd, capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        return {"error": f"Git clone failed: {result.stderr.strip()}"}

    # Parse manifest
    manifest = _parse_skill_manifest(tmp_dir)
    if manifest is None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"error": "No valid SKILL.md manifest found (requires name, version, description in YAML front matter)"}

    skill_name = manifest["name"]

    # Validate all Python files
    errors = _validate_all_skills(tmp_dir)
    if errors:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"error": f"Skill validation failed: {'; '.join(errors)}"}

    # Move to final location
    final_dir = marketplace_dir / skill_name
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.move(str(tmp_dir), str(final_dir))

    # Remove .git directory to save space
    git_dir = final_dir / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)

    # Write install metadata
    metadata = {
        "name": skill_name,
        "version": manifest.get("version", "unknown"),
        "description": manifest.get("description", ""),
        "repo_url": repo_url,
        "ref": ref,
    }
    (final_dir / ".installed.json").write_text(json.dumps(metadata, indent=2) + "\n")

    logger.info("Installed marketplace skill: %s v%s", skill_name, metadata["version"])
    return {"installed": True, **metadata}


def list_skills(marketplace_dir: Path) -> list[dict]:
    """List all installed marketplace skills."""
    skills: list[dict] = []
    if not marketplace_dir.exists():
        return skills

    for entry in sorted(marketplace_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        meta_file = entry / ".installed.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                skills.append(meta)
            except (json.JSONDecodeError, OSError):
                skills.append({"name": entry.name, "version": "unknown"})
        else:
            skills.append({"name": entry.name, "version": "unknown"})

    return skills


def remove_skill(name: str, marketplace_dir: Path) -> dict:
    """Remove an installed marketplace skill."""
    skill_dir = marketplace_dir / name
    if not skill_dir.exists():
        return {"error": f"Skill '{name}' not found"}

    shutil.rmtree(skill_dir)
    logger.info("Removed marketplace skill: %s", name)
    return {"removed": True, "name": name}
