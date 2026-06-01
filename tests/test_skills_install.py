"""Tests for SKILL.md install/distribution: marketplace, operator tools, endpoint."""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import patch

from src.agent.builtins import skill_admin_tool
from src.agent.skills import SkillStore
from src.marketplace import _clone_repo, install_skill, remove_skill

SKILL_MD = "---\nname: demo-skill\ndescription: A demo skill\nversion: 2.0.0\n---\n# Body\nDo it.\n"


def _mock_clone_writing(text: str):
    """Return a subprocess.run mock that populates the clone target with a SKILL.md."""
    def _run(cmd, **kwargs):
        target = Path(cmd[-1])
        target.mkdir(parents=True, exist_ok=True)
        (target / "SKILL.md").write_text(text, encoding="utf-8")
        return type("R", (), {"returncode": 0, "stderr": ""})()
    return _run


# ── _clone_repo validation (no git invoked) ───────────────────────────────

def test_clone_repo_rejects_bad_scheme(tmp_path):
    assert "https://" in (_clone_repo("file:///etc", tmp_path / "d") or "")
    assert "https://" in (_clone_repo("http://x/y.git", tmp_path / "d") or "")


def test_clone_repo_rejects_flag_ref(tmp_path):
    err = _clone_repo("https://x/y.git", tmp_path / "d", ref="--upload-pack=evil")
    assert err and "ref" in err


# ── install_skill ──────────────────────────────────────────────────────────

def test_install_skill_success(tmp_path):
    skills_dir = tmp_path / "skills_installed"
    with patch("src.marketplace.subprocess.run", side_effect=_mock_clone_writing(SKILL_MD)):
        result = install_skill("https://github.com/u/demo.git", skills_dir)
    assert result["installed"] is True
    assert result["name"] == "demo-skill"
    assert result["version"] == "2.0.0"
    assert (skills_dir / "demo-skill" / "SKILL.md").exists()
    assert (skills_dir / "demo-skill" / ".installed.json").exists()
    # tmp staging cleaned up
    assert not (skills_dir / "_tmp_install").exists()


def test_install_skill_no_manifest(tmp_path):
    def _run(cmd, **kwargs):
        target = Path(cmd[-1])
        target.mkdir(parents=True, exist_ok=True)
        (target / "README.md").write_text("no skill here", encoding="utf-8")
        return type("R", (), {"returncode": 0, "stderr": ""})()

    with patch("src.marketplace.subprocess.run", side_effect=_run):
        result = install_skill("https://github.com/u/x.git", tmp_path / "s")
    assert "error" in result and "SKILL.md" in result["error"]


def test_install_skill_clone_failure(tmp_path):
    def _run(cmd, **kwargs):
        return type("R", (), {"returncode": 128, "stderr": "not found"})()

    with patch("src.marketplace.subprocess.run", side_effect=_run):
        result = install_skill("https://github.com/u/x.git", tmp_path / "s")
    assert "error" in result and "clone failed" in result["error"].lower()


def test_install_skill_bad_scheme(tmp_path):
    result = install_skill("file:///etc/passwd", tmp_path / "s")
    assert "error" in result


def test_installed_skill_visible_via_store(tmp_path):
    skills_dir = tmp_path / "skills_installed"
    with patch("src.marketplace.subprocess.run", side_effect=_mock_clone_writing(SKILL_MD)):
        install_skill("https://github.com/u/demo.git", skills_dir)
    store = SkillStore(bundled_dir=tmp_path / "bundled", installed_dir=skills_dir)
    skill = store.get("demo-skill")
    assert skill is not None
    assert skill.source == "installed"


# ── remove_skill ────────────────────────────────────────────────────────────

def test_remove_skill_success(tmp_path):
    skills_dir = tmp_path / "skills_installed"
    with patch("src.marketplace.subprocess.run", side_effect=_mock_clone_writing(SKILL_MD)):
        install_skill("https://github.com/u/demo.git", skills_dir)
    result = remove_skill("demo-skill", skills_dir)
    assert result["removed"] is True
    assert not (skills_dir / "demo-skill").exists()


def test_remove_skill_not_found(tmp_path):
    assert "error" in remove_skill("nope", tmp_path / "s")


def test_remove_skill_invalid_name(tmp_path):
    assert "error" in remove_skill("../escape", tmp_path / "s")


# ── operator tools: gating ───────────────────────────────────────────────────

class _FakeMesh:
    def __init__(self):
        self.installed = None
        self.removed = None

    async def install_skill(self, repo_url, ref=""):
        self.installed = (repo_url, ref)
        return {"installed": True, "name": "demo-skill"}

    async def remove_skill(self, name):
        self.removed = name
        return {"removed": True, "name": name}


async def test_install_tool_requires_operator(monkeypatch):
    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    result = await skill_admin_tool.install_skill("https://x/y.git", mesh_client=_FakeMesh())
    assert "only available to the operator" in result["error"]


async def test_install_tool_requires_user_origin(monkeypatch):
    monkeypatch.setenv("ALLOWED_TOOLS", "install_skill")
    monkeypatch.setattr("src.agent.loop._last_message_is_user_origin", lambda m: False)
    result = await skill_admin_tool.install_skill(
        "https://x/y.git", mesh_client=_FakeMesh(), _messages=[{"role": "user"}],
    )
    assert result["error"] == "provenance_check_failed"


async def test_install_tool_forwards_to_mesh(monkeypatch):
    monkeypatch.setenv("ALLOWED_TOOLS", "install_skill")
    monkeypatch.setattr("src.agent.loop._last_message_is_user_origin", lambda m: True)
    mesh = _FakeMesh()
    result = await skill_admin_tool.install_skill(
        "https://x/y.git", ref="main", mesh_client=mesh, _messages=[{"role": "user"}],
    )
    assert result["installed"] is True
    assert mesh.installed == ("https://x/y.git", "main")


async def test_remove_tool_forwards_to_mesh(monkeypatch):
    monkeypatch.setenv("ALLOWED_TOOLS", "remove_skill")
    monkeypatch.setattr("src.agent.loop._last_message_is_user_origin", lambda m: True)
    mesh = _FakeMesh()
    result = await skill_admin_tool.remove_skill(
        "demo-skill", mesh_client=mesh, _messages=[{"role": "user"}],
    )
    assert result["removed"] is True
    assert mesh.removed == "demo-skill"


# ── mesh endpoint smoke (wiring + success through the route) ─────────────────

def _make_app(container_manager=None):
    import tempfile

    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app
    from src.host.traces import TraceStore

    d = Path(tempfile.mkdtemp())
    perms = PermissionMatrix()
    app = create_mesh_app(
        blackboard=Blackboard(db_path=str(d / "bb.db")),
        pubsub=PubSub(),
        router=MessageRouter(perms, {}),
        permissions=perms,
        cost_tracker=CostTracker(str(d / "c.db")),
        trace_store=TraceStore(str(d / "t.db")),
        container_manager=container_manager,
    )
    return app


async def test_endpoint_503_without_container_manager():
    from httpx import ASGITransport, AsyncClient

    app = _make_app(container_manager=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.post("/mesh/skills/install", json={"repo_url": "https://x/y.git"})
    assert resp.status_code == 503


async def test_endpoint_400_missing_repo_url(tmp_path):
    from httpx import ASGITransport, AsyncClient

    cm = types.SimpleNamespace(project_root=tmp_path)
    app = _make_app(container_manager=cm)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.post("/mesh/skills/install", json={})
    assert resp.status_code == 400


async def test_endpoint_install_success(tmp_path):
    from httpx import ASGITransport, AsyncClient

    cm = types.SimpleNamespace(project_root=tmp_path)
    app = _make_app(container_manager=cm)
    with patch("src.marketplace.subprocess.run", side_effect=_mock_clone_writing(SKILL_MD)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.post(
                "/mesh/skills/install", json={"repo_url": "https://github.com/u/demo.git"},
            )
    assert resp.status_code == 200, resp.text
    assert resp.json()["name"] == "demo-skill"
    assert (tmp_path / "skills_installed" / "demo-skill" / "SKILL.md").exists()
