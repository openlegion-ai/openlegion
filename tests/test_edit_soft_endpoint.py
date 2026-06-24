"""Tests for the PR 1 soft-edit endpoint: POST /mesh/agents/{id}/edit-soft."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.shared.types import AgentPermissions, MessageOrigin


def _human_origin_headers(agent_id: str = "operator") -> dict:
    origin = MessageOrigin(kind="human", channel="cli", user="u1")
    return {
        "X-Agent-ID": agent_id,
        "X-Origin": origin.to_header_value(),
    }


def _agent_origin_headers(agent_id: str = "operator") -> dict:
    origin = MessageOrigin(kind="agent", channel="", user="")
    return {
        "X-Agent-ID": agent_id,
        "X-Origin": origin.to_header_value(),
    }


def _agents_yaml(tmp_path, names: list[str]) -> Path:
    (tmp_path / "config").mkdir(exist_ok=True)
    afile = tmp_path / "config" / "agents.yaml"
    cfg = {"agents": {
        name: {
            "role": name,
            "model": "openai/gpt-4o-mini",
            "initial_instructions": f"# old instructions for {name}",
            "initial_soul": f"# old soul for {name}",
        }
        for name in names
    }}
    afile.write_text(yaml.dump(cfg))
    return afile


@pytest.fixture
def mesh_app(tmp_path, monkeypatch):
    """Mesh app pinned to a tmp_path with `writer` agent on disk.

    No credential_vault is wired — matches the production behavior
    where edit_agent_soft's BYOK model validator is a no-op when no
    vault exists (test harnesses / sandbox transport). The dedicated
    validator tests below construct their own vault-wired app.
    """
    afile = _agents_yaml(tmp_path, names=["writer", "researcher"])
    monkeypatch.chdir(tmp_path)
    import src.cli.config as cli_cfg
    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", afile)
    monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", tmp_path / "config" / "permissions.json")
    (tmp_path / "config" / "permissions.json").write_text('{"permissions": {}}')

    import src.host.server as server_module
    importlib.reload(server_module)

    perms_map = {
        "operator": {"can_route_tasks": True, "can_manage_projects": True},
        "writer": {"can_route_tasks": False},
        "researcher": {"can_route_tasks": False},
    }
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    for aid, p in perms_map.items():
        permissions.permissions[aid] = AgentPermissions(agent_id=aid, **p)
    router = MessageRouter(permissions, {})
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
    )
    yield app, server_module, tmp_path
    blackboard.close()
    importlib.reload(server_module)


@pytest.mark.asyncio
async def test_edit_soft_happy_path_writes_yaml_and_returns_undo(mesh_app):
    """Soft-edit on instructions writes YAML and returns an undo_token."""
    app, server_module, tmp_path = mesh_app
    afile = tmp_path / "config" / "agents.yaml"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={
                "field": "instructions",
                "value": "# new punchier instructions",
                "reason": "user_asked",
            },
            headers=_human_origin_headers(),
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["agent_id"] == "writer"
    assert body["field"] == "instructions"
    assert body["undo_token"]
    assert body["expires_at"]
    assert "writer" in body["summary"]

    # YAML was actually rewritten with the new value.
    cfg = yaml.safe_load(afile.read_text())
    assert (
        cfg["agents"]["writer"]["initial_instructions"]
        == "# new punchier instructions"
    )

    # And the change_history store has the row keyed on the returned token.
    assert app.change_history.peek(body["undo_token"]) is not None


@pytest.mark.asyncio
async def test_edit_soft_accepts_hard_field_model_with_30min_ttl(mesh_app):
    """Hard fields (model/permissions/budget/thinking) now apply
    immediately via /edit-soft and emit a receipt with a 30-min undo
    window. The propose+confirm gate was retired in favor of "all
    edits apply immediately, undo is the safety net".
    """
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={"field": "model", "value": "anthropic/claude-haiku", "reason": "user_asked"},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["agent_id"] == "writer"
    assert body["field"] == "model"
    assert body["undo_token"]
    # 30-minute TTL on the receipt for hard fields.
    assert body["ttl_seconds"] == 1800
    assert body["field_class"] == "hard"


@pytest.mark.asyncio
async def test_edit_soft_accepts_max_output_tokens_and_writes_yaml(mesh_app):
    """The per-agent output cap is a hard field: applies immediately, gets a
    30-min undo window, and persists to YAML so it survives restart."""
    app, _, tmp_path = mesh_app
    afile = tmp_path / "config" / "agents.yaml"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={"field": "max_output_tokens", "value": 32000, "reason": "user_asked"},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["field"] == "max_output_tokens"
    assert body["field_class"] == "hard"
    assert body["ttl_seconds"] == 1800
    cfg = yaml.safe_load(afile.read_text())
    assert cfg["agents"]["writer"]["max_output_tokens"] == 32000

    # When the cap was never set before, the recorded "before" value is the
    # effective default (16384), not "". This makes the audit sensible and —
    # critically — makes Undo restore an int that the hot-reload push can
    # forward to the live agent (the push guard requires an int), instead of
    # silently leaving the running container on the raised cap.
    change = app.change_history.peek(body["undo_token"])
    assert change is not None
    assert change["old_value"] == 16384
    assert change["new_value"] == 32000


@pytest.mark.asyncio
async def test_edit_soft_rejects_out_of_range_max_output_tokens(mesh_app):
    """Out-of-range / non-integer caps are rejected server-side with 400,
    mirroring the operator-tool guard and the agent /config endpoint."""
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        for bad in (200_001, "8192"):
            r = await c.post(
                "/mesh/agents/writer/edit-soft",
                json={"field": "max_output_tokens", "value": bad, "reason": "user_asked"},
                headers=_human_origin_headers(),
            )
            assert r.status_code == 400, r.text
            assert "max_output_tokens" in r.json()["detail"]


@pytest.mark.asyncio
async def test_edit_soft_accepts_hard_field_permissions(mesh_app):
    """Permissions edits apply immediately with a 30-min undo window."""
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={
                "field": "permissions",
                "value": {"can_use_browser": True},
                "reason": "user_asked",
            },
            headers=_human_origin_headers(),
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["field_class"] == "hard"
    assert body["ttl_seconds"] == 1800
    assert body["undo_token"]


@pytest.mark.asyncio
async def test_edit_soft_rejects_can_use_wallet_grant(mesh_app):
    """H1: the mesh edit-soft endpoint re-enforces the operator ceiling.

    A fooled / injected operator LLM (or any non-dashboard caller) that
    POSTs a raw permissions edit granting ``can_use_wallet=True`` must be
    rejected server-side with HTTP 400 even though the client-side
    operator-tool guard was bypassed.
    """
    app, _, tmp_path = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={
                "field": "permissions",
                "value": {"can_use_wallet": True},
                "reason": "user_asked",
            },
            headers=_human_origin_headers(),
        )
    assert r.status_code == 400, r.text
    assert "ceiling" in r.json()["detail"].lower()

    # The escalation must NOT have been persisted.
    import json as _json
    perms = _json.loads((tmp_path / "config" / "permissions.json").read_text())
    assert perms["permissions"].get("writer", {}).get("can_use_wallet") is not True


@pytest.mark.asyncio
async def test_edit_soft_allows_can_spawn_grant(mesh_app):
    """can_spawn is a default-on capability now (no longer ceiling-blocked),
    so the operator may grant it via edit-soft — only can_use_wallet remains
    server-side ceiling-blocked (the test above)."""
    app, _, tmp_path = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={
                "field": "permissions",
                "value": {"can_spawn": True},
                "reason": "user_asked",
            },
            headers=_human_origin_headers(),
        )
    assert r.status_code == 200, r.text

    import json as _json
    perms = _json.loads((tmp_path / "config" / "permissions.json").read_text())
    assert perms["permissions"].get("writer", {}).get("can_spawn") is True


@pytest.mark.asyncio
async def test_edit_soft_rejects_out_of_ceiling_blackboard_write(mesh_app):
    """H1: blackboard_write patterns outside the ceiling are rejected."""
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={
                "field": "permissions",
                "value": {"blackboard_write": ["secrets/*"]},
                "reason": "user_asked",
            },
            headers=_human_origin_headers(),
        )
    assert r.status_code == 400, r.text
    assert "ceiling" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_edit_soft_within_ceiling_permissions_still_succeeds(mesh_app):
    """H1 regression guard: a within-ceiling permissions edit still applies.

    The server-side ceiling re-check must not block legitimate edits
    (``can_use_browser=True`` + an allowed ``blackboard_write`` pattern).
    """
    app, _, tmp_path = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={
                "field": "permissions",
                "value": {
                    "can_use_browser": True,
                    "blackboard_write": ["artifacts/*"],
                },
                "reason": "user_asked",
            },
            headers=_human_origin_headers(),
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["field"] == "permissions"
    assert body["undo_token"]

    import json as _json
    perms = _json.loads((tmp_path / "config" / "permissions.json").read_text())
    assert perms["permissions"]["writer"]["can_use_browser"] is True


@pytest.mark.asyncio
async def test_edit_soft_soft_field_keeps_5min_ttl(mesh_app):
    """Soft fields keep the snappy 5-minute undo window."""
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={"field": "instructions", "value": "# tightened", "reason": "user_asked"},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["field_class"] == "soft"
    assert body["ttl_seconds"] == 300


@pytest.mark.asyncio
async def test_edit_soft_rejects_invalid_model_value(mesh_app):
    """Validation still runs — empty model rejected with 400."""
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={"field": "model", "value": "", "reason": "user_asked"},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 400
    assert "model" in r.text.lower()


def _vault_wired_mesh_app(tmp_path, monkeypatch, provider_keys: dict[str, str]):
    """Build a mesh app with a live credential_vault. ``provider_keys``
    maps provider name → key value, e.g. ``{"anthropic": "sk-ant-x"}``.
    Returns ``(app, server_module)``."""
    afile = _agents_yaml(tmp_path, names=["writer"])
    monkeypatch.chdir(tmp_path)
    import src.cli.config as cli_cfg
    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", afile)
    monkeypatch.setattr(
        cli_cfg, "PERMISSIONS_FILE", tmp_path / "config" / "permissions.json",
    )
    (tmp_path / "config" / "permissions.json").write_text('{"permissions": {}}')

    # Plant ONLY the requested provider keys; drop the others so the
    # vault reports exactly the providers under test.
    for prov in ("openai", "anthropic", "google", "deepseek", "xai"):
        monkeypatch.delenv(f"OPENLEGION_SYSTEM_{prov.upper()}_API_KEY", raising=False)
    for prov, key in provider_keys.items():
        monkeypatch.setenv(f"OPENLEGION_SYSTEM_{prov.upper()}_API_KEY", key)

    import src.host.server as server_module
    importlib.reload(server_module)
    from src.host.credentials import CredentialVault

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    for aid, p in (
        ("operator", {"can_route_tasks": True, "can_manage_projects": True}),
        ("writer", {"can_route_tasks": False}),
    ):
        permissions.permissions[aid] = AgentPermissions(agent_id=aid, **p)
    router = MessageRouter(permissions, {})
    vault = CredentialVault()  # picks up env via _load_credentials()

    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        credential_vault=vault,
    )
    return app, server_module, blackboard


@pytest.mark.asyncio
async def test_edit_soft_rejects_model_without_provider_credentials(
    tmp_path, monkeypatch,
):
    """BYOK validation closes the /edit-soft back-door: if the operator
    tries to retarget an agent at a model whose provider has no API
    key configured, the edit must 400 — otherwise the agent silently
    dies on its next LLM call (same failure mode PR #901 plugged at
    create-agent time).

    Uses a vault-wired app so the validator runs (it's a no-op when
    credential_vault is None — matches create-agent's gating).
    """
    app, _, bb = _vault_wired_mesh_app(
        tmp_path, monkeypatch,
        provider_keys={"anthropic": "sk-ant-only"},  # NO openai
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t",
        ) as c:
            r = await c.post(
                "/mesh/agents/writer/edit-soft",
                json={"field": "model", "value": "openai/gpt-4o", "reason": "u"},
                headers=_human_origin_headers(),
            )
        assert r.status_code == 400, r.text
        body = r.text.lower()
        assert "openai" in body
        assert "credentials" in body or "key" in body
    finally:
        bb.close()
        import src.host.server as server_module
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_edit_soft_accepts_model_when_provider_has_credentials(
    tmp_path, monkeypatch,
):
    """Positive case: vault has an anthropic key, so /edit-soft to an
    anthropic/* model is accepted."""
    app, _, bb = _vault_wired_mesh_app(
        tmp_path, monkeypatch,
        provider_keys={"anthropic": "sk-ant-only"},
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t",
        ) as c:
            r = await c.post(
                "/mesh/agents/writer/edit-soft",
                json={
                    "field": "model",
                    "value": "anthropic/claude-sonnet-4-20250514",
                    "reason": "u",
                },
                headers=_human_origin_headers(),
            )
        assert r.status_code == 200, r.text
        assert r.json()["field"] == "model"
    finally:
        bb.close()
        import src.host.server as server_module
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_edit_soft_skips_validation_when_no_vault_wired(mesh_app):
    """Production parity guard: when ``create_mesh_app`` is built without
    a ``credential_vault`` (test harnesses, sandbox transport), the
    model-field validator must be a no-op. Otherwise existing
    vault-less tests that POST arbitrary model strings would all
    regress to 400. Mirrors ``create_custom_agent``'s gating.
    """
    app, _, _ = mesh_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t",
    ) as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            # A provider that almost certainly has no env key in CI;
            # without the no-vault skip this would 400.
            json={"field": "model", "value": "cohere/command-x", "reason": "u"},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_edit_soft_rejects_invalid_thinking_value(mesh_app):
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={"field": "thinking", "value": "ultra", "reason": "user_asked"},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_edit_soft_rejects_empty_field(mesh_app):
    """Empty `field` payload must 400 with 'field is required' before any
    YAML or audit write — guards against caller bugs that drop the
    field key. Pre-existing guard, now reachable by hard-field callers
    too via the unified endpoint."""
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={"field": "", "value": "ignored"},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 400
    assert "field is required" in r.text.lower()


@pytest.mark.asyncio
async def test_edit_soft_rejects_invalid_field(mesh_app):
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={"field": "made_up", "value": "x"},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_edit_soft_blocks_self_modification(mesh_app):
    """Operator cannot soft-edit itself even server-side."""
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/operator/edit-soft",
            json={"field": "instructions", "value": "x"},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_edit_soft_404_unknown_agent(mesh_app):
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/ghost/edit-soft",
            json={"field": "instructions", "value": "x"},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_edit_soft_403_for_non_operator(mesh_app):
    """A worker agent must not be able to call edit-soft."""
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={"field": "instructions", "value": "x"},
            headers={"X-Agent-ID": "researcher"},  # not operator
        )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_edit_soft_accepts_agent_origin(mesh_app):
    """Soft edits explicitly do NOT require human origin (the receipt+undo
    is the safety net). Agent-origin should still apply."""
    app, _, _ = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={"field": "soul", "value": "be calm"},
            headers=_agent_origin_headers(),
        )
    assert r.status_code == 200, r.text
    assert r.json()["undo_token"]


@pytest.mark.asyncio
async def test_edit_soft_emits_receipt_event(mesh_app):
    """The endpoint must emit operator_action_receipt on the event bus
    so the dashboard can render the inline card."""
    app, server_module, tmp_path = mesh_app
    # Wire a recording event bus.
    events = []

    class _Bus:
        def emit(self, event_type, agent="", data=None):
            events.append((event_type, agent, data))

    app.change_history.set_event_bus(_Bus())
    # The endpoint emits via the locally-captured event_bus reference,
    # so we patch that too. Since create_mesh_app captures event_bus at
    # closure time, rebuild the app with our bus instead.
    blackboard = Blackboard(str(tmp_path / "bb2.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    permissions.permissions["operator"] = AgentPermissions(
        agent_id="operator", can_route_tasks=True,
    )
    router = MessageRouter(permissions, {})
    bus = _Bus()
    app2 = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        event_bus=bus,
    )
    try:
        async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://t") as c:
            r = await c.post(
                "/mesh/agents/writer/edit-soft",
                json={"field": "interface", "value": "Accepts X, returns Y."},
                headers=_human_origin_headers(),
            )
        assert r.status_code == 200, r.text
        receipt_events = [e for e in events if e[0] == "operator_action_receipt"]
        assert len(receipt_events) == 1
        _, agent, data = receipt_events[0]
        assert agent == "operator"
        assert data["agent_id"] == "writer"
        assert data["field"] == "interface"
        assert data["undo_token"]
        assert data["new_value"] == "Accepts X, returns Y."
    finally:
        blackboard.close()


@pytest.mark.asyncio
async def test_edit_soft_hard_field_emits_agent_config_updated(
    mesh_app, monkeypatch,
):
    """Hard-field edits MUST also emit ``agent_config_updated`` so the
    dashboard's agent config card flips to the new value live. Soft
    fields are covered by ``operator_action_receipt``; hard fields fire
    BOTH events (receipt for chat-card rendering + config_updated for
    the SPA's agent-detail panel). ``live`` reports whether the
    agent-side hot-reload (model / thinking) succeeded — True here
    because no agent container is registered with the test mesh, so the
    push-to-agent branch is skipped and ``hot_reload_ok`` stays True.
    """
    app, server_module, tmp_path = mesh_app
    events: list[tuple[str, str, dict]] = []

    class _Bus:
        def emit(self, event_type, agent="", data=None):
            events.append((event_type, agent, data))

    blackboard = Blackboard(str(tmp_path / "bb_cfg.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    permissions.permissions["operator"] = AgentPermissions(
        agent_id="operator", can_route_tasks=True,
    )
    router = MessageRouter(permissions, {})
    bus = _Bus()
    app2 = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        event_bus=bus,
    )
    try:
        async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://t") as c:
            r = await c.post(
                "/mesh/agents/writer/edit-soft",
                json={
                    "field": "permissions",
                    "value": {"can_use_browser": True},
                    "reason": "user_asked",
                },
                headers=_human_origin_headers(),
            )
        assert r.status_code == 200, r.text
        cfg_events = [e for e in events if e[0] == "agent_config_updated"]
        assert len(cfg_events) == 1
        _, agent, data = cfg_events[0]
        assert agent == "writer"
        assert data["agent_id"] == "writer"
        assert data["field"] == "permissions"
        # No new_value on the wire — permissions diffs are sensitive.
        assert "new_value" not in data and "old_value" not in data
        # ``live`` reports hot-reload status. No agent registered, so
        # the push-to-agent branch is skipped and ``hot_reload_ok``
        # stays True (default for fields without an agent-side reload).
        assert data["live"] is True
    finally:
        blackboard.close()


@pytest.mark.asyncio
async def test_edit_soft_supersede_emits_marker_for_prior_receipts(mesh_app):
    """A second soft-edit on the same field must mark the prior receipt
    as superseded so the dashboard can warn the operator before they
    click [Undo] (which would silently discard the newer edit)."""
    app, server_module, tmp_path = mesh_app
    events: list[tuple[str, str, dict]] = []

    class _Bus:
        def emit(self, event_type, agent="", data=None):
            events.append((event_type, agent, data))

    blackboard = Blackboard(str(tmp_path / "bb3.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    permissions.permissions["operator"] = AgentPermissions(
        agent_id="operator", can_route_tasks=True,
    )
    router = MessageRouter(permissions, {})
    bus = _Bus()
    app2 = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        event_bus=bus,
    )
    try:
        async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://t") as c:
            r1 = await c.post(
                "/mesh/agents/writer/edit-soft",
                json={"field": "instructions", "value": "v1"},
                headers=_human_origin_headers(),
            )
            assert r1.status_code == 200, r1.text
            assert r1.json()["supersedes_count"] == 0
            tok1 = r1.json()["undo_token"]

            r2 = await c.post(
                "/mesh/agents/writer/edit-soft",
                json={"field": "instructions", "value": "v2"},
                headers=_human_origin_headers(),
            )
            assert r2.status_code == 200, r2.text
            # The second edit reports it supersedes one prior receipt.
            assert r2.json()["supersedes_count"] == 1

        # Two receipts emitted, plus a superseded marker for the first.
        kinds = [e[0] for e in events]
        assert kinds.count("operator_action_receipt") == 2
        superseded = [e for e in events if e[0] == "operator_action_receipt_superseded"]
        assert len(superseded) == 1
        assert superseded[0][2]["undo_token"] == tok1
        assert superseded[0][2]["agent_id"] == "writer"
        assert superseded[0][2]["field"] == "instructions"

        # A different field should NOT trigger a supersede event.
        events.clear()
        async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://t") as c:
            r3 = await c.post(
                "/mesh/agents/writer/edit-soft",
                json={"field": "soul", "value": "calm"},
                headers=_human_origin_headers(),
            )
            assert r3.status_code == 200
        assert not any(e[0] == "operator_action_receipt_superseded" for e in events)
    finally:
        blackboard.close()


@pytest.mark.asyncio
async def test_edit_soft_records_audit_log(mesh_app):
    """The shared _apply_pending_change writes to the audit log; soft-edit
    must inherit that so the activity feed has a record."""
    app, _, tmp_path = mesh_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            "/mesh/agents/writer/edit-soft",
            json={"field": "role", "value": "Senior Writer"},
            headers=_human_origin_headers(),
        )
    assert r.status_code == 200
    # Audit log via the blackboard exposed as app.blackboard? Not directly —
    # but we can verify via the change_history row.
    rows = app.change_history.list_recent(agent_id="writer")
    assert any(r["field"] == "role" for r in rows)


# ── Seam follow-up Fix 4: edit_agent(model=...) clears quarantine ──


@pytest.mark.asyncio
async def test_edit_soft_model_clears_quarantine(tmp_path, monkeypatch):
    """A successful field=model edit is the operator's "fix the credential"
    signal — quarantine clears implicitly so the lane resumes dispatching.
    No separate clear_quarantine operator tool needed."""
    from unittest.mock import MagicMock

    from src.host.health import HealthMonitor

    afile = _agents_yaml(tmp_path, names=["writer"])
    monkeypatch.chdir(tmp_path)
    import src.cli.config as cli_cfg
    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", afile)
    monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", tmp_path / "config" / "permissions.json")
    (tmp_path / "config" / "permissions.json").write_text('{"permissions": {}}')

    import src.host.server as server_module
    importlib.reload(server_module)

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    permissions.permissions["operator"] = AgentPermissions(
        agent_id="operator", can_route_tasks=True, can_manage_projects=True,
    )
    permissions.permissions["writer"] = AgentPermissions(agent_id="writer")
    router = MessageRouter(permissions, {})

    health_monitor = HealthMonitor(
        runtime=MagicMock(), transport=MagicMock(), router=router,
    )
    health_monitor.register("writer")
    # Force quarantine.
    for _ in range(3):
        health_monitor.record_auth_failure(
            "writer", provider="openai", model="x", http_status=401,
        )
    assert health_monitor.is_quarantined("writer") is True

    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        health_monitor=health_monitor,
    )
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post(
                "/mesh/agents/writer/edit-soft",
                json={
                    "field": "model",
                    "value": "anthropic/claude-haiku",
                    "reason": "fix credential",
                },
                headers=_human_origin_headers(),
            )
        assert r.status_code == 200, r.text
        # Quarantine should have been cleared by the edit.
        assert health_monitor.is_quarantined("writer") is False
        assert health_monitor.agents["writer"].consecutive_auth_failures == 0
    finally:
        blackboard.close()
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_edit_soft_non_model_field_does_not_clear_quarantine(
    tmp_path, monkeypatch,
):
    """Only model changes clear quarantine — editing instructions doesn't,
    because that doesn't fix a broken credential."""
    from unittest.mock import MagicMock

    from src.host.health import HealthMonitor

    afile = _agents_yaml(tmp_path, names=["writer"])
    monkeypatch.chdir(tmp_path)
    import src.cli.config as cli_cfg
    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", afile)
    monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", tmp_path / "config" / "permissions.json")
    (tmp_path / "config" / "permissions.json").write_text('{"permissions": {}}')

    import src.host.server as server_module
    importlib.reload(server_module)

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    permissions = PermissionMatrix()
    permissions.permissions["operator"] = AgentPermissions(
        agent_id="operator", can_route_tasks=True, can_manage_projects=True,
    )
    permissions.permissions["writer"] = AgentPermissions(agent_id="writer")
    router = MessageRouter(permissions, {})

    health_monitor = HealthMonitor(
        runtime=MagicMock(), transport=MagicMock(), router=router,
    )
    health_monitor.register("writer")
    for _ in range(3):
        health_monitor.record_auth_failure(
            "writer", provider="openai", model="x", http_status=401,
        )

    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        health_monitor=health_monitor,
    )
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post(
                "/mesh/agents/writer/edit-soft",
                json={
                    "field": "instructions",
                    "value": "# new instructions",
                    "reason": "u",
                },
                headers=_human_origin_headers(),
            )
        assert r.status_code == 200, r.text
        # Quarantine remains — instructions don't fix credentials.
        assert health_monitor.is_quarantined("writer") is True
    finally:
        blackboard.close()
        importlib.reload(server_module)
