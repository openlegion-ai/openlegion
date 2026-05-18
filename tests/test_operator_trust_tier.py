"""Tests for the operator trust tier.

Covers M1 (structured denial logging + caller-side envelope) and M2
(two-tier trust model + four-gate short-circuits + fail-closed startup).

The trust tier introduces a deliberate asymmetry:

- Operator bypasses coordination/scope gates (can_message, blackboard,
  publish/subscribe, route_tasks). Workers stay subject.
- Operator stays gated on real-blast-radius surfaces — wallet, credential
  value reads, vault management, browser-drive actions. Those check
  operator's own grant like any other agent.

The carve-out is what makes the trust tier safe: anything the user
controls at the chat/UI layer (revoke browser, revoke creds, budget
caps, destructive-action nonces) stays enforced. Anything that only
gates inter-agent coordination is the wrong layer to defend against
the user's own operator and is what produced the silent-stall incidents
this work is closing.
"""

from __future__ import annotations

import importlib
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from src.host.costs import CostTracker
from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import AgentPermissions, PermissionMatrix
from src.host.traces import TraceStore


def _reload_server():
    """Reload ``src.host.server`` so module-level constants re-evaluate."""
    import src.host.server as server_module
    importlib.reload(server_module)
    return server_module


@pytest.fixture
def mesh_setup(tmp_path, monkeypatch):
    """Construct a mesh app with auth tokens and minimal permissions.

    Operator's grants are deliberately MINIMAL — empty can_message, empty
    blackboard ACL, no can_route_tasks — so any successful coordination
    call exercises the trust-tier short-circuit rather than the grant.
    Workers receive default-deny grants for the same reason: if the test
    asserts a worker gets 403, that proves the regression guard.
    """
    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "warn")
    server = _reload_server()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "operator": AgentPermissions(
            agent_id="operator",
            # Intentionally narrow: trust tier must override these.
            can_message=[],
            can_spawn=False,
            can_manage_fleet=False,
            blackboard_read=[],
            blackboard_write=[],
        ),
        "trend-scout": AgentPermissions(
            agent_id="trend-scout",
            can_message=["seo-strategist"],  # narrow but valid
            blackboard_read=["projects/x/*"],
            blackboard_write=["projects/x/*"],
        ),
        "seo-strategist": AgentPermissions(
            agent_id="seo-strategist",
            can_message=[],
            blackboard_read=[],
            blackboard_write=[],
        ),
    }
    perms._config_path = str(tmp_path / "perms.json")

    router = MessageRouter(permissions=perms, agent_registry={})
    router.register_agent("operator", "http://operator:8400", [])
    router.register_agent("trend-scout", "http://trend-scout:8400", [])
    router.register_agent("seo-strategist", "http://seo-strategist:8400", [])

    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    auth_tokens = {
        "operator": "operator-secret",
        "trend-scout": "scout-secret",
        "seo-strategist": "strat-secret",
    }

    app = server.create_mesh_app(
        blackboard=bb,
        pubsub=pubsub,
        router=router,
        permissions=perms,
        cost_tracker=costs,
        trace_store=traces,
        auth_tokens=auth_tokens,
    )

    yield {
        "app": app,
        "bb": bb,
        "perms": perms,
        "router": router,
        "server": server,
        "tokens": auth_tokens,
    }

    bb.close()
    costs.close()
    traces.close()
    monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
    _reload_server()


def _hdr(token: str) -> dict:
    return {"authorization": f"Bearer {token}"}


# =============================================================================
# Boot-time fail-closed gate
# =============================================================================


def test_enforce_mode_without_tokens_in_production_raises_systemexit(
    tmp_path, monkeypatch,
):
    """Production-posture boot: enforce + no tokens + no bypass-var → SystemExit.

    Conftest sets ``OPENLEGION_SKIP_TRUST_TIER_BOOT_GATE=1`` for the
    test session so other fixtures can boot under enforce+no-tokens
    safely. To exercise the production path we unset that var for the
    duration of the call.
    """
    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "enforce")
    monkeypatch.delenv("OPENLEGION_SKIP_TRUST_TIER_BOOT_GATE", raising=False)
    server = _reload_server()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})

    with pytest.raises(SystemExit, match="auth_tokens"):
        server.create_mesh_app(
            blackboard=bb, pubsub=pubsub, router=router,
            permissions=perms, auth_tokens={},
        )

    bb.close()
    monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
    _reload_server()


def test_enforce_mode_with_tokens_boots_clean(tmp_path, monkeypatch):
    """Enforce + populated tokens is the documented production posture."""
    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "enforce")
    server = _reload_server()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})

    app = server.create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router,
        permissions=perms, auth_tokens={"operator": "x"},
    )
    assert app is not None

    bb.close()
    monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
    _reload_server()


@pytest.mark.parametrize(
    "bypass_value",
    ["", "0", "true", "True", "yes", " 1", "1 ", "false", "anything-else"],
)
def test_bypass_var_only_accepts_exact_string_one(
    tmp_path, monkeypatch, bypass_value,
):
    """Strict-value check: only ``"1"`` exactly bypasses.

    Regression guard against a future refactor flipping the check to a
    truthy form (``if os.environ.get(...):``) which would silently
    accept ``"true"`` / ``"yes"`` / whitespace and weaken the gate.
    """
    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "enforce")
    monkeypatch.setenv("OPENLEGION_SKIP_TRUST_TIER_BOOT_GATE", bypass_value)
    server = _reload_server()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})

    with pytest.raises(SystemExit, match="auth_tokens"):
        server.create_mesh_app(
            blackboard=bb, pubsub=pubsub, router=router,
            permissions=perms, auth_tokens={},
        )

    bb.close()
    monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
    monkeypatch.delenv("OPENLEGION_SKIP_TRUST_TIER_BOOT_GATE", raising=False)
    _reload_server()


def test_test_fixtures_exempt_from_fail_closed(tmp_path, monkeypatch):
    """Regression guard: the conftest bypass-var must let fixtures boot.

    The first draft of this gate broke 135 existing tests that
    legitimately run under enforce mode without tokens. The
    ``OPENLEGION_SKIP_TRUST_TIER_BOOT_GATE=1`` bypass (set globally in
    ``tests/conftest.py``) must keep those fixtures green.
    """
    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "enforce")
    # Conftest already sets the bypass var — assert it's in effect.
    import os
    assert os.environ.get("OPENLEGION_SKIP_TRUST_TIER_BOOT_GATE") == "1"
    server = _reload_server()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})

    # No tokens, bypass var present — must boot cleanly.
    app = server.create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router,
        permissions=perms, auth_tokens={},
    )
    assert app is not None

    bb.close()
    monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
    _reload_server()


# =============================================================================
# Helper unit
# =============================================================================


def test_caller_is_operator_helper_returns_true_for_operator():
    server = _reload_server()
    fake_request = MagicMock()
    assert server._caller_is_operator("operator", fake_request) is True


def test_caller_is_operator_helper_returns_false_for_worker():
    server = _reload_server()
    fake_request = MagicMock()
    assert server._caller_is_operator("trend-scout", fake_request) is False
    assert server._caller_is_operator("", fake_request) is False


# =============================================================================
# Gate bypass — operator coordinates without per-grant approval
# =============================================================================


@pytest.mark.asyncio
async def test_operator_wake_bypasses_can_message(mesh_setup):
    """Operator wake must NOT 403 even when can_message is empty.

    This is the bug-report headline: trend-scout → seo-strategist 403
    on hand_off. Operator running the same flow must reach past the
    gate. The fixture has no lane_manager, so the wake handler's
    fallback router path may raise downstream (which the test simply
    treats as "past the gate" — not a 403). What matters is that the
    permissions.can_message check did not deny.
    """
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        try:
            resp = await client.post(
                "/mesh/wake",
                params={"target": "seo-strategist", "message": "go"},
                headers=_hdr(mesh_setup["tokens"]["operator"]),
            )
            # Past the gate — could be 200 (delivered), 500 (downstream
            # fixture quirk), 404 (registry miss). Just must NOT be 403.
            assert resp.status_code != 403, resp.text
        except Exception as e:
            # Some httpx + ASGITransport edge cases re-raise an inner
            # validation error before the response can be built. Any
            # non-HTTPException is fine — the gate would have produced
            # an HTTPException(403) cleanly.
            assert "403" not in str(e), str(e)


@pytest.mark.asyncio
async def test_worker_wake_still_gated_by_can_message(mesh_setup):
    """Regression guard: a worker with an empty can_message must still 403."""
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        try:
            # seo-strategist has empty can_message — must NOT be able to wake.
            resp = await client.post(
                "/mesh/wake",
                params={"target": "trend-scout", "message": "x"},
                headers=_hdr(mesh_setup["tokens"]["seo-strategist"]),
            )
        except Exception as e:  # noqa: BLE001
            pytest.fail(
                f"Worker wake should 403 cleanly — got pre-gate exception: {e}"
            )
    assert resp.status_code == 403
    assert "cannot wake" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_operator_blackboard_read_bypasses_grant(mesh_setup):
    """Operator with empty blackboard_read still reads."""
    app = mesh_setup["app"]
    # Seed a key.
    mesh_setup["bb"].write("projects/x/k1", {"v": 1}, written_by="trend-scout")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/mesh/blackboard/projects/x/k1",
            params={"agent_id": "operator"},
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_operator_blackboard_write_bypasses_grant(mesh_setup):
    """Operator with empty blackboard_write still writes."""
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.put(
            "/mesh/blackboard/projects/x/k2",
            params={"agent_id": "operator"},
            json={"v": 2},
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_worker_blackboard_write_still_gated(mesh_setup):
    """Regression guard: worker without write grant gets 403."""
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        # seo-strategist has empty blackboard_write.
        resp = await client.put(
            "/mesh/blackboard/projects/x/k3",
            params={"agent_id": "seo-strategist"},
            json={"v": 3},
            headers=_hdr(mesh_setup["tokens"]["seo-strategist"]),
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_operator_publish_bypasses_can_publish(mesh_setup):
    """Operator publishes to any topic without can_publish grant.

    No subscribers → publish returns ``{subscribers_notified: 0}`` — the
    point is that the gate didn't fire.
    """
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/publish",
            json={"source": "operator", "topic": "any/topic", "payload": {}},
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    assert resp.status_code != 403, resp.text


@pytest.mark.asyncio
async def test_operator_message_bypasses_can_message(mesh_setup):
    """``/mesh/message`` accepts operator → any-target even with empty grant.

    Distinct from ``/mesh/wake``: wake is the followup-lane notify path;
    /mesh/message is the standard router-routed message. Both check
    ``can_message``; both must bypass for operator.
    """
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        try:
            resp = await client.post(
                "/mesh/message",
                json={
                    "from_agent": "operator", "to": "seo-strategist",
                    "type": "task_request", "payload": {"x": 1},
                },
                headers=_hdr(mesh_setup["tokens"]["operator"]),
            )
            assert resp.status_code != 403, resp.text
        except Exception as e:  # noqa: BLE001
            # Downstream routing requires real transport; the gate's
            # the only thing under test here.
            assert "403" not in str(e)


@pytest.mark.asyncio
async def test_operator_subscribe_bypasses_can_subscribe(mesh_setup):
    """Subscribing operator to any topic must not 403 on the grant gate."""
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post(
            "/mesh/subscribe",
            params={"topic": "any/topic", "agent_id": "operator"},
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    assert resp.status_code != 403, resp.text


@pytest.mark.asyncio
async def test_operator_route_tasks_bypasses_grant(mesh_setup):
    """``POST /mesh/tasks`` accepts operator without can_route_tasks grant."""
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        try:
            resp = await client.post(
                "/mesh/tasks",
                json={
                    "assignee": "seo-strategist",
                    "title": "Test task",
                    "description": "operator-routed work",
                },
                headers=_hdr(mesh_setup["tokens"]["operator"]),
            )
            assert resp.status_code != 403, resp.text
        except Exception as e:  # noqa: BLE001
            assert "403" not in str(e)


# =============================================================================
# Still gated — operator has no special pass on real-blast-radius surfaces
# =============================================================================


@pytest.mark.asyncio
async def test_operator_wallet_still_gated(mesh_setup):
    """Operator wallet access checks operator's own grant — no bypass.

    Wallet operations are irreversible (crypto). The trust tier
    deliberately excludes them — the user-control layer (budget caps,
    chain allowlists) is the right place to gate operator's reach, not
    a blanket short-circuit at the mesh layer.
    """
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get(
            "/mesh/wallet/address",
            params={"agent_id": "operator", "chain": "ethereum"},
            headers=_hdr(mesh_setup["tokens"]["operator"]),
        )
    # Operator's permissions do not include can_use_wallet — must 403.
    assert resp.status_code == 403


def test_operator_still_gated_surfaces_not_in_bypass_grep(mesh_setup):
    """Static-inspection regression: every carve-out gate must NOT be
    wrapped in a ``_caller_is_operator`` short-circuit.

    Hitting these endpoints via the HTTP layer requires fully-wired
    credential vault, wallet service, and browser service fixtures, which
    would balloon the test surface beyond the trust tier. Instead, grep
    the source: a bypass added by accident to one of these gates would
    show up here.

    Contract this test pins (fragile, but a deliberate trip-wire):
    - The ``_caller_is_operator(...)`` short-circuit lives within the 5
      lines immediately above the corresponding ``permissions.can_*``
      check. If a future refactor moves bypass to a same-line clause
      (``if _caller_is_operator(...) or not permissions.can_use_wallet(...)``)
      or to a helper variable assigned earlier in the function, this
      test silently passes. Keep the bypass flush against the gate.
    - Code comments mentioning the gate name are excluded from the scan
      (a comment like ``# this is NOT can_use_wallet`` near an unrelated
      bypass would otherwise false-positive).

    The wallet-route test above proves the principle end-to-end at the
    HTTP layer; this test pins the static-inspection invariant for the
    full carve-out family.
    """
    server_path = mesh_setup["server"].__file__
    with open(server_path) as f:
        source_lines = f.readlines()

    # Full carve-out: every gate operator MUST NOT bypass. Wallet ops are
    # irreversible (crypto); vault management exposes cred metadata;
    # cred-value reads expose secrets; ``can_use_browser`` and
    # ``can_browser_action`` gate real browser drive actions (operator
    # coordinates the fleet but doesn't drive the browser).
    gated_gates = [
        "permissions.can_use_wallet(",
        "permissions.can_use_wallet_chain(",
        "permissions.can_access_wallet_contract(",
        "permissions.can_manage_vault(",
        "permissions.can_access_credential(",
        "permissions.can_use_browser(",
        "permissions.can_browser_action(",
    ]
    for gate in gated_gates:
        gate_lines = []
        for i, line in enumerate(source_lines):
            if gate not in line:
                continue
            # Skip non-call contexts: def lines, imports, and comments
            # (a comment naming the gate near unrelated code shouldn't
            # trip the scan).
            stripped = line.lstrip()
            if stripped.startswith(("def ", "import ", "#")):
                continue
            gate_lines.append((i, line))
        assert gate_lines, f"Could not find any callsite for {gate!r}"
        for idx, line in gate_lines:
            # Walk back 5 lines for a ``_caller_is_operator`` wrap.
            preceding = "".join(source_lines[max(0, idx - 5):idx])
            assert "_caller_is_operator" not in preceding, (
                f"Unexpected operator bypass before {gate!r} at line {idx + 1}. "
                f"Real-blast-radius gates must check operator's own grant."
            )


# =============================================================================
# Forgery rejection — bearer-derived identity wins over X-Agent-ID header
# =============================================================================


@pytest.mark.asyncio
async def test_x_agent_id_operator_with_worker_token_does_not_bypass(mesh_setup):
    """Forging ``X-Agent-ID: operator`` with a worker bearer is ignored.

    The bearer is HMAC-compared against ``_auth_tokens``; the identity
    derived from that comparison wins. ``X-Agent-ID`` is only honored on
    the loopback internal path, which this request is NOT on.
    """
    app = mesh_setup["app"]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        try:
            # seo-strategist's bearer but claims to be operator via header.
            resp = await client.post(
                "/mesh/wake",
                params={"target": "trend-scout", "message": "x"},
                headers={
                    "authorization": f"Bearer {mesh_setup['tokens']['seo-strategist']}",
                    "X-Agent-ID": "operator",  # claim ignored
                },
            )
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"Should 403 cleanly — got pre-gate exception: {e}")
    # Caller resolves to seo-strategist (empty can_message) — must 403.
    assert resp.status_code == 403
    assert "seo-strategist" in resp.json()["detail"]


# =============================================================================
# Structured denial logging (M1)
# =============================================================================


def test_record_denial_emits_structured_log_fields(caplog):
    """``_record_denial`` must include caller/target/gate in both the
    formatted message AND the structured ``extra`` dict.

    JSON-mode production logs serialize the LogRecord's attributes —
    the formatted message is human-readable, but JSON consumers depend
    on the canonical keys ``denial_category`` / ``denial_caller`` /
    ``denial_target`` / ``denial_gate`` being attached to the record.
    """
    import logging

    server = _reload_server()
    caplog.set_level(logging.WARNING, logger="host.server")
    server._record_denial(
        "permission", caller="trend-scout", target="projects/x/k",
        gate="blackboard.write:can_write_blackboard",
    )

    # Message-format assertion (human-readable).
    matched = [
        r for r in caplog.records
        if "mesh denial" in r.getMessage()
        and "trend-scout" in r.getMessage()
        and "blackboard.write" in r.getMessage()
    ]
    assert matched, [r.getMessage() for r in caplog.records]

    # Structured-extra assertion (machine-readable). The ``extra=``
    # kwarg on ``logger.warning`` sets each key as an attribute on
    # the LogRecord, so JSON formatters pick them up directly.
    record = matched[0]
    assert record.denial_category == "permission"
    assert record.denial_caller == "trend-scout"
    assert record.denial_target == "projects/x/k"
    assert record.denial_gate == "blackboard.write:can_write_blackboard"


def test_record_denial_unknown_category_is_silent():
    """Unknown categories must be silently ignored — no crash, no counter bump."""
    server = _reload_server()
    server._denial_counter.clear()
    server._record_denial("not-a-real-category", caller="x")
    assert "not-a-real-category" not in server._denial_counter


def test_record_denial_increments_counter():
    """The 24h counter must still increment for known categories."""
    server = _reload_server()
    server._denial_counter.clear()
    server._record_denial(
        "scope", caller="trend-scout", target="t", gate="publish:project_prefix",
    )
    assert server._denial_counter["scope"] == 1


# =============================================================================
# Caller-side envelope (M1) — _raise_with_body
# =============================================================================


def test_raise_with_body_includes_detail_from_json():
    """JSON ``detail`` body lands in the exception string."""
    from src.agent.mesh_client import _raise_with_body

    request = httpx.Request("POST", "http://mesh:8420/mesh/wake")
    response = httpx.Response(
        status_code=403,
        request=request,
        content=json.dumps({"detail": "Agent trend-scout cannot wake seo-strategist"}).encode(),
        headers={"content-type": "application/json"},
    )
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        _raise_with_body(response)
    assert "Agent trend-scout cannot wake seo-strategist" in str(exc_info.value)


def test_raise_with_body_includes_text_body_when_not_json():
    """Non-JSON 403 body still surfaces (truncated to 500 chars)."""
    from src.agent.mesh_client import _raise_with_body

    request = httpx.Request("GET", "http://mesh:8420/mesh/blackboard/k")
    response = httpx.Response(
        status_code=403,
        request=request,
        content=b"plain-text denial body",
    )
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        _raise_with_body(response)
    assert "plain-text denial body" in str(exc_info.value)


def test_raise_with_body_passthrough_on_success():
    """2xx responses must not raise."""
    from src.agent.mesh_client import _raise_with_body

    request = httpx.Request("GET", "http://mesh:8420/mesh/agents")
    response = httpx.Response(status_code=200, request=request, content=b"{}")
    # No exception.
    _raise_with_body(response)
