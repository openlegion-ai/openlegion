"""Task 4 — operator authentication & registration.

Tests the cryptographic gate on ``POST /mesh/register`` for the reserved
``agent_id="operator"`` claim, plus the unchanged rejections for ``mesh``
and ``canary-probe``. Identity is anchored to the per-agent bearer token
pool that ``src/host/runtime.py`` already generates; no new env var.

See ``docs/plans/archive/2026-05-02-operator-orchestration-roadmap.md`` Task 4
for the full rationale.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _build_mesh_app(tmp_path, auth_tokens: dict[str, str] | None):
    """Construct a minimal mesh app for the registration tests.

    Mirrors the harness used by ``tests/test_permissions.py`` and
    ``tests/test_mesh.py``: real ``Blackboard`` + ``PubSub`` +
    ``MessageRouter`` + ``PermissionMatrix``, with optional
    ``auth_tokens`` to exercise the production posture where bearer
    verification is on.
    """
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app

    perms = PermissionMatrix()
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    router = MessageRouter(perms, {})

    app = create_mesh_app(
        blackboard=bb,
        pubsub=pubsub,
        router=router,
        permissions=perms,
        auth_tokens=auth_tokens,
    )
    return app, router, bb


def test_operator_registration_with_correct_bearer_succeeds(tmp_path):
    """Real operator container → /mesh/register with its token → 200.

    The operator's bearer matches ``auth_tokens["operator"]`` so the
    cryptographic gate passes. Subsequent registry inspection confirms
    the agent landed in ``router.agent_registry`` under ``"operator"``.
    """
    auth_tokens = {"operator": "tok-op", "scout": "tok-scout"}
    app, router, bb = _build_mesh_app(tmp_path, auth_tokens)
    try:
        client = TestClient(app)
        resp = client.post(
            "/mesh/register",
            json={"agent_id": "operator", "capabilities": ["chat"], "port": 8400},
            headers={"Authorization": "Bearer tok-op"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"registered": True}
        assert "operator" in router.agent_registry
    finally:
        bb.close()


def test_worker_bearer_cannot_register_as_operator(tmp_path):
    """Worker bearer + agent_id='operator' body → 403.

    Without Task 4 the request would silently succeed: ``_resolve_agent_id``
    would override the body's claim with the bearer's identity and the
    worker would register as ``scout``. Task 4 catches the spoofed claim
    loudly so a buggy or hostile caller can't slip through.
    """
    auth_tokens = {"operator": "tok-op", "scout": "tok-scout"}
    app, router, bb = _build_mesh_app(tmp_path, auth_tokens)
    try:
        client = TestClient(app)
        resp = client.post(
            "/mesh/register",
            json={"agent_id": "operator", "capabilities": [], "port": 8400},
            headers={"Authorization": "Bearer tok-scout"},
        )
        assert resp.status_code == 403, resp.text
        assert "operator" not in router.agent_registry
        # Error message must not echo the bearer value.
        assert "tok-scout" not in resp.text
    finally:
        bb.close()


def test_wrong_bearer_cannot_register_as_operator(tmp_path):
    """A bearer that doesn't match any registered agent → 403."""
    auth_tokens = {"operator": "tok-op", "scout": "tok-scout"}
    app, router, bb = _build_mesh_app(tmp_path, auth_tokens)
    try:
        client = TestClient(app)
        resp = client.post(
            "/mesh/register",
            json={"agent_id": "operator", "capabilities": [], "port": 8400},
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 403, resp.text
        assert "operator" not in router.agent_registry
        assert "wrong" not in resp.text
    finally:
        bb.close()


def test_missing_authorization_header_cannot_register_as_operator(tmp_path):
    """No Authorization header → 403 even with auth tokens configured."""
    auth_tokens = {"operator": "tok-op"}
    app, router, bb = _build_mesh_app(tmp_path, auth_tokens)
    try:
        client = TestClient(app)
        resp = client.post(
            "/mesh/register",
            json={"agent_id": "operator", "capabilities": [], "port": 8400},
        )
        assert resp.status_code == 403, resp.text
        assert "operator" not in router.agent_registry
    finally:
        bb.close()


def test_wrong_bearer_different_lengths_both_fail(tmp_path):
    """A bearer of a different length than the operator's still fails.

    Defensive coverage that constant-time compare doesn't accidentally
    short-circuit on length mismatch in a way that observably differs
    from the same-length wrong-bearer path. Both must produce 403.
    """
    auth_tokens = {"operator": "tok-op-12345"}  # 12 chars
    app, router, bb = _build_mesh_app(tmp_path, auth_tokens)
    try:
        client = TestClient(app)
        # Different-length wrong bearer.
        resp1 = client.post(
            "/mesh/register",
            json={"agent_id": "operator", "capabilities": [], "port": 8400},
            headers={"Authorization": "Bearer x"},
        )
        # Same-length wrong bearer.
        resp2 = client.post(
            "/mesh/register",
            json={"agent_id": "operator", "capabilities": [], "port": 8400},
            headers={"Authorization": "Bearer wrongtokenAB"},  # 13 chars vs 12
        )
        assert resp1.status_code == 403
        assert resp2.status_code == 403
        assert "operator" not in router.agent_registry
    finally:
        bb.close()


def test_mesh_agent_id_still_rejected_with_operator_bearer(tmp_path):
    """Even with the operator's bearer, ``agent_id='mesh'`` is rejected.

    ``mesh`` is a system-only reserved identity used internally by the
    router for synthetic messages; it must never come back over the
    register endpoint regardless of caller authority.
    """
    auth_tokens = {"operator": "tok-op"}
    app, router, bb = _build_mesh_app(tmp_path, auth_tokens)
    try:
        client = TestClient(app)
        resp = client.post(
            "/mesh/register",
            json={"agent_id": "mesh", "capabilities": [], "port": 8400},
            headers={"Authorization": "Bearer tok-op"},
        )
        assert resp.status_code == 403, resp.text
        assert "mesh" not in router.agent_registry
    finally:
        bb.close()


def test_canary_probe_agent_id_still_rejected_with_operator_bearer(tmp_path):
    """``canary-probe`` continues to use its internal-only path.

    Even the operator's bearer cannot register ``canary-probe`` over
    HTTP; the stealth canary sweeper plugs into the router directly via
    ``router.register_agent``.
    """
    auth_tokens = {"operator": "tok-op"}
    app, router, bb = _build_mesh_app(tmp_path, auth_tokens)
    try:
        client = TestClient(app)
        resp = client.post(
            "/mesh/register",
            json={"agent_id": "canary-probe", "capabilities": [], "port": 8400},
            headers={"Authorization": "Bearer tok-op"},
        )
        assert resp.status_code == 403, resp.text
        assert "canary-probe" not in router.agent_registry
    finally:
        bb.close()


def test_empty_token_pool_rejects_operator_registration(tmp_path):
    """Fail-closed: no auth_tokens configured → operator path is rejected.

    Per Task 4 plan: ``If auth_tokens is empty (test mode without token
    pool), the operator path falls through to the 'not allowed' branch.``
    Operator identity must be cryptographic, never positional — even in
    dev/test we don't allow 'first to call wins'.
    """
    app, router, bb = _build_mesh_app(tmp_path, auth_tokens=None)
    try:
        client = TestClient(app)
        # Even without a bearer we must reject — there is no oracle to
        # match against, so the operator path is structurally closed.
        resp = client.post(
            "/mesh/register",
            json={"agent_id": "operator", "capabilities": [], "port": 8400},
        )
        assert resp.status_code == 403, resp.text
        assert "operator" not in router.agent_registry
    finally:
        bb.close()


def test_empty_token_pool_with_bearer_still_rejects_operator(tmp_path):
    """Even with a bearer header, empty token pool → operator path fails."""
    app, router, bb = _build_mesh_app(tmp_path, auth_tokens={})
    try:
        client = TestClient(app)
        resp = client.post(
            "/mesh/register",
            json={"agent_id": "operator", "capabilities": [], "port": 8400},
            headers={"Authorization": "Bearer anything"},
        )
        assert resp.status_code == 403, resp.text
        assert "operator" not in router.agent_registry
    finally:
        bb.close()


def test_register_uses_constant_time_comparison():
    """The registration gate must use ``hmac.compare_digest``.

    Static guarantee: the helper imports ``hmac`` and the register
    handler uses ``hmac.compare_digest`` to compare the supplied bearer
    against ``auth_tokens["operator"]``. We assert this by reading the
    source so a future refactor can't accidentally fall back to ``==``.
    """
    import inspect

    from src.host import server as server_module

    src = inspect.getsource(server_module.create_mesh_app)
    # The register endpoint compares the bearer using compare_digest.
    # Find the register_agent function source within create_mesh_app
    # and assert hmac.compare_digest appears in the operator branch.
    assert "hmac.compare_digest(bearer" in src, (
        "Operator registration gate must use hmac.compare_digest "
        "for timing-safe bearer comparison"
    )


def test_operator_registration_round_trip(tmp_path):
    """After a successful operator register, the agent appears in the
    registry and downstream endpoints accept it.

    Sanity-check that we didn't break the post-register side effects:
    ``router.agent_registry["operator"]`` is populated and
    ``GET /mesh/agents`` includes it.
    """
    auth_tokens = {"operator": "tok-op"}
    app, router, bb = _build_mesh_app(tmp_path, auth_tokens)
    try:
        client = TestClient(app)
        # Register
        resp = client.post(
            "/mesh/register",
            json={"agent_id": "operator", "capabilities": ["chat"], "port": 8400},
            headers={"Authorization": "Bearer tok-op"},
        )
        assert resp.status_code == 200, resp.text
        assert "operator" in router.agent_registry

        # List agents — operator must be visible.
        resp2 = client.get(
            "/mesh/agents",
            headers={"Authorization": "Bearer tok-op"},
        )
        assert resp2.status_code == 200, resp2.text
        agents = resp2.json()
        assert "operator" in agents
        # Operator carries the global scope marker.
        assert agents["operator"].get("scope") == "global"
    finally:
        bb.close()


def test_normal_agent_registration_unaffected(tmp_path):
    """Worker registration with a matching bearer still works.

    Regression guard: the new operator-specific branch must not break
    the standard ``_resolve_agent_id`` path that workers go through.
    """
    auth_tokens = {"scout": "tok-scout"}
    app, router, bb = _build_mesh_app(tmp_path, auth_tokens)
    try:
        client = TestClient(app)
        resp = client.post(
            "/mesh/register",
            json={"agent_id": "scout", "capabilities": ["search"], "port": 8401},
            headers={"Authorization": "Bearer tok-scout"},
        )
        assert resp.status_code == 200, resp.text
        assert "scout" in router.agent_registry
    finally:
        bb.close()


@pytest.mark.parametrize("invalid_id", ["", "../etc/passwd", "a" * 100])
def test_invalid_agent_id_format_rejected(tmp_path, invalid_id):
    """Format check still applies to the new branching code."""
    auth_tokens = {"operator": "tok-op"}
    app, router, bb = _build_mesh_app(tmp_path, auth_tokens)
    try:
        client = TestClient(app)
        resp = client.post(
            "/mesh/register",
            json={"agent_id": invalid_id, "capabilities": [], "port": 8400},
            headers={"Authorization": "Bearer tok-op"},
        )
        assert resp.status_code == 400, resp.text
    finally:
        bb.close()


# === src/host/runtime.py — operator-specific token wiring =================

def test_docker_backend_operator_gets_distinct_token(tmp_path, monkeypatch):
    """``DockerBackend`` registers each agent (including operator) with a
    distinct token in ``auth_tokens``. No fleet-shared token leaks.

    Uses a constructed-but-not-running backend to call ``start_agent``
    only up to the token assignment — Docker calls are mocked.
    """
    import secrets as _secrets
    from unittest.mock import MagicMock

    from src.host.runtime import DockerBackend

    # Build a backend without invoking __init__ (avoids docker SDK init).
    backend = DockerBackend.__new__(DockerBackend)
    backend.auth_tokens = {}
    backend._next_port = 9000
    import threading
    backend._port_lock = threading.Lock()
    backend.use_host_network = False
    backend.mesh_host_port = 8420
    backend.extra_env = {}
    backend.project_root = tmp_path
    backend.uploads_dir = tmp_path / "uploads"
    backend.uploads_dir.mkdir(exist_ok=True)
    backend._network_name = "openlegion_agents"
    backend.BASE_IMAGE = "openlegion-agent:latest"
    backend.client = MagicMock()
    backend.agents = {}

    # Mock the docker client so start_agent doesn't try to talk to Docker.
    fake_container = MagicMock()
    fake_container.id = "abc123"
    backend.client.containers.run.return_value = fake_container
    backend.client.containers.get.side_effect = Exception("not found")

    # Stub out Docker import inside start_agent.
    fake_docker = MagicMock()
    fake_docker.errors.NotFound = Exception
    monkeypatch.setitem(__import__("sys").modules, "docker", fake_docker)

    # Use a deterministic token generator so we can spot duplication.
    seq = iter(["tok-op", "tok-scout", "tok-extra"])
    monkeypatch.setattr(_secrets, "token_urlsafe", lambda n: next(seq))
    # The runtime imports secrets at module level — patch the module attr.
    import src.host.runtime as runtime_mod
    monkeypatch.setattr(runtime_mod, "secrets", _secrets)

    # Start operator and a worker.
    backend.start_agent(
        agent_id="operator", role="orchestrator", tools_dir="",
        env_overrides={"ALLOWED_TOOLS": "chat"},
    )
    backend.start_agent(
        agent_id="scout", role="worker", tools_dir="",
    )

    # Each agent has its own distinct token.
    assert backend.auth_tokens["operator"] == "tok-op"
    assert backend.auth_tokens["scout"] == "tok-scout"
    assert backend.auth_tokens["operator"] != backend.auth_tokens["scout"]

    # Operator's container env was given the operator-specific token.
    op_call = backend.client.containers.run.call_args_list[0]
    op_env = op_call.kwargs["environment"]
    assert op_env["MESH_AUTH_TOKEN"] == "tok-op"
    assert op_env["AGENT_ID"] == "operator"
    # Scout's container env was given the scout-specific token (not op's).
    scout_call = backend.client.containers.run.call_args_list[1]
    scout_env = scout_call.kwargs["environment"]
    assert scout_env["MESH_AUTH_TOKEN"] == "tok-scout"
    assert scout_env["MESH_AUTH_TOKEN"] != op_env["MESH_AUTH_TOKEN"]


def test_sandbox_backend_operator_gets_distinct_token(tmp_path):
    """``SandboxBackend._prepare_workspace`` writes the operator's
    operator-specific token into ``.agent.env`` and registers it under
    ``auth_tokens["operator"]``. Workers get distinct tokens.
    """
    from src.host.runtime import SandboxBackend

    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "PROJECT.md").write_text("# Test")

    backend = SandboxBackend.__new__(SandboxBackend)
    backend.project_root = project_root
    backend.mesh_host_port = 8420
    backend.agents = {}
    backend.auth_tokens = {}
    backend.extra_env = {}
    backend._workspace_root = tmp_path / ".openlegion" / "agents"
    backend._workspace_root.mkdir(parents=True)

    # Operator workspace.
    op_ws = backend._prepare_workspace(
        agent_id="operator", role="orchestrator", tools_dir="",
        system_prompt="", model="",
        env_overrides={"ALLOWED_TOOLS": "chat"},
    )
    # Worker workspace.
    sw_ws = backend._prepare_workspace(
        agent_id="scout", role="worker", tools_dir="",
        system_prompt="", model="",
    )

    op_token = backend.auth_tokens["operator"]
    sw_token = backend.auth_tokens["scout"]
    assert op_token and sw_token
    assert op_token != sw_token

    op_env_text = (op_ws / ".agent.env").read_text()
    sw_env_text = (sw_ws / ".agent.env").read_text()
    assert f"MESH_AUTH_TOKEN={op_token}" in op_env_text
    assert f"MESH_AUTH_TOKEN={sw_token}" in sw_env_text
    # The operator's token does NOT appear in the worker's env file.
    assert op_token not in sw_env_text


def test_runtime_restart_rotates_operator_token(tmp_path):
    """A fresh ``SandboxBackend`` instance generates a new operator token.

    Each runtime startup rotates per-agent tokens; the operator is no
    different. Without rotation, a leaked token from a prior run would
    grant indefinite access.
    """
    from src.host.runtime import SandboxBackend

    project_root = tmp_path / "project"
    project_root.mkdir()

    def _new_backend(workspace_root: str) -> SandboxBackend:
        b = SandboxBackend.__new__(SandboxBackend)
        b.project_root = project_root
        b.mesh_host_port = 8420
        b.agents = {}
        b.auth_tokens = {}
        b.extra_env = {}
        wr = tmp_path / workspace_root
        wr.mkdir(parents=True, exist_ok=True)
        b._workspace_root = wr
        return b

    b1 = _new_backend("run1")
    b1._prepare_workspace(
        agent_id="operator", role="orch", tools_dir="",
        system_prompt="", model="",
    )
    token1 = b1.auth_tokens["operator"]

    b2 = _new_backend("run2")
    b2._prepare_workspace(
        agent_id="operator", role="orch", tools_dir="",
        system_prompt="", model="",
    )
    token2 = b2.auth_tokens["operator"]

    # Token rotation: new runtime → new token.
    assert token1 != token2
    assert token1 and token2  # both non-empty


def test_post_rotation_register_uses_new_token(tmp_path):
    """After rotation, /mesh/register requires the NEW token.

    Cross-component check: token1 (from a prior run) must NOT register
    as operator in the new run; token2 (the current run's token) must.
    """
    from src.host.runtime import SandboxBackend

    project_root = tmp_path / "project"
    project_root.mkdir()
    (tmp_path / "ws").mkdir(exist_ok=True)

    backend = SandboxBackend.__new__(SandboxBackend)
    backend.project_root = project_root
    backend.mesh_host_port = 8420
    backend.agents = {}
    backend.auth_tokens = {}
    backend.extra_env = {}
    backend._workspace_root = tmp_path / "ws"

    backend._prepare_workspace(
        agent_id="operator", role="orch", tools_dir="",
        system_prompt="", model="",
    )
    current_token = backend.auth_tokens["operator"]
    stale_token = "tok-from-previous-run"

    (tmp_path / "mesh").mkdir(exist_ok=True)
    app, router, bb = _build_mesh_app(
        tmp_path / "mesh", auth_tokens=dict(backend.auth_tokens),
    )
    try:
        client = TestClient(app)

        # Stale token rejected.
        resp_stale = client.post(
            "/mesh/register",
            json={"agent_id": "operator", "capabilities": [], "port": 8400},
            headers={"Authorization": f"Bearer {stale_token}"},
        )
        assert resp_stale.status_code == 403

        # Current rotated token accepted.
        resp_curr = client.post(
            "/mesh/register",
            json={"agent_id": "operator", "capabilities": [], "port": 8400},
            headers={"Authorization": f"Bearer {current_token}"},
        )
        assert resp_curr.status_code == 200
    finally:
        bb.close()
