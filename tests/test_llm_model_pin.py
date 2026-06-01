"""Tests for the H3 LLM-proxy model pin.

Finding H3: the LLM proxy gated only on ``can_use_api("llm")`` while the
agent fully controlled ``params["model"]`` — so a cheap-model agent could
route through ANY configured provider key (cost drain / key abuse). The
fix pins the REQUESTED model to the agent's configured model (read from
``config/settings.json`` — the same row written at create/edit time).

Scoping that matters:
  - The pin applies ONLY to the agent-REQUESTED model, never to a
    failover substitute the mesh chooses internally — so legitimate
    failover (where the vault returns a *different* ``model`` than asked)
    is never 403'd.
  - The pin auto-updates when the operator edits the agent's model
    (edit rewrites the same config row ``_load_config`` reads).
"""

from __future__ import annotations

import importlib

import pytest
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import AgentPermissions, PermissionMatrix
from src.shared.types import APIProxyResponse


def _reload_server():
    import src.host.server as server_module
    importlib.reload(server_module)
    return server_module


class _FakeVault:
    """Minimal credential-vault stand-in for the proxy path.

    ``execute_api_call`` echoes back whatever ``used_model`` it's told to
    (so a test can simulate a failover substitution where the returned
    model differs from the requested one). ``is_model_compatible`` is a
    permissive allowlist so we test the *pin*, not compatibility.
    """

    def __init__(self, used_model: str | None = None, compatible: bool = True):
        self._used_model = used_model
        self._compatible = compatible
        self.calls: list[dict] = []

    async def execute_api_call(self, request, agent_id: str = ""):
        requested = request.params.get("model", "")
        self.calls.append({"agent_id": agent_id, "model": requested})
        return APIProxyResponse(
            success=True,
            data={
                "content": "ok",
                "tokens_used": 1,
                "input_tokens": 1,
                "output_tokens": 0,
                # Simulate the mesh's internal failover substituting a
                # different model than the agent requested.
                "model": self._used_model or requested,
                "tool_calls": [],
            },
        )

    def is_model_compatible(self, model: str):
        if self._compatible:
            return (True, None)
        return (False, f"Model '{model}' is not compatible.")


@pytest.fixture
def pin_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "warn")
    server = _reload_server()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "operator": AgentPermissions(agent_id="operator"),
        # Worker is granted can_use_api("llm") so the request reaches the
        # pin gate (the pin is the surface under test, not can_use_api).
        "writer": AgentPermissions(agent_id="writer", allowed_apis=["llm"]),
    }
    perms._config_path = str(tmp_path / "perms.json")

    router = MessageRouter(permissions=perms, agent_registry={})
    router.register_agent("operator", "http://operator:8400", [])
    router.register_agent("writer", "http://writer:8400", [])

    auth_tokens = {"operator": "operator-secret", "writer": "writer-secret"}

    # The pin reads the agent's configured model from this config dict.
    # Default: writer is pinned to the cheap model.
    cfg_holder = {
        "cfg": {
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "agents": {"writer": {"model": "openai/gpt-4o-mini"}},
        },
    }
    monkeypatch.setattr(
        "src.cli.config._load_config", lambda *a, **k: cfg_holder["cfg"],
    )

    vault = _FakeVault()

    app = server.create_mesh_app(
        blackboard=bb,
        pubsub=pubsub,
        router=router,
        permissions=perms,
        credential_vault=vault,
        auth_tokens=auth_tokens,
    )

    yield {
        "app": app,
        "vault": vault,
        "cfg_holder": cfg_holder,
        "tokens": auth_tokens,
    }

    bb.close()
    monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
    _reload_server()


def _hdr(token: str) -> dict:
    return {"authorization": f"Bearer {token}"}


def _req(model: str) -> dict:
    return {
        "service": "llm",
        "action": "chat",
        "params": {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        },
        "timeout": 30,
    }


async def _post(app, body, token, agent_id):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/mesh/api",
            json=body,
            params={"agent_id": agent_id},
            headers=_hdr(token),
        )


# (1) Different provider/model than configured → 403
@pytest.mark.asyncio
async def test_requesting_other_model_is_403(pin_setup):
    app = pin_setup["app"]
    resp = await _post(
        app, _req("anthropic/claude-opus-4"), "writer-secret", "writer",
    )
    assert resp.status_code == 403, resp.text
    assert "not authorized to use model" in resp.text
    # Never reached the vault.
    assert pin_setup["vault"].calls == []


# (2) The agent's own configured model → allowed
@pytest.mark.asyncio
async def test_requesting_configured_model_allowed(pin_setup):
    app = pin_setup["app"]
    resp = await _post(
        app, _req("openai/gpt-4o-mini"), "writer-secret", "writer",
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True
    assert pin_setup["vault"].calls[-1]["model"] == "openai/gpt-4o-mini"


# (3) Editing the agent's model updates the pin (config row is the source
#     of truth — the pin reads it live, so a swap flips which model passes)
@pytest.mark.asyncio
async def test_edit_updates_the_pin(pin_setup):
    app = pin_setup["app"]
    # Before edit: opus is rejected.
    resp = await _post(
        app, _req("anthropic/claude-opus-4"), "writer-secret", "writer",
    )
    assert resp.status_code == 403

    # Operator "edits" the model — same effect as edit-soft rewriting the
    # config row that _load_config returns.
    pin_setup["cfg_holder"]["cfg"]["agents"]["writer"]["model"] = (
        "anthropic/claude-opus-4"
    )

    # After edit: opus is now allowed, old cheap model is rejected.
    resp = await _post(
        app, _req("anthropic/claude-opus-4"), "writer-secret", "writer",
    )
    assert resp.status_code == 200, resp.text

    resp = await _post(
        app, _req("openai/gpt-4o-mini"), "writer-secret", "writer",
    )
    assert resp.status_code == 403


# (4) Internal failover still works: the agent requests its configured
#     model, the vault substitutes a *different* model internally — the
#     pin must NOT 403 the substitute (it only gates the requested model).
@pytest.mark.asyncio
async def test_failover_substitute_not_blocked(pin_setup):
    app = pin_setup["app"]
    # Vault returns a different model than requested (failover happened
    # deep inside _call_llm_with_failover).
    pin_setup["vault"]._used_model = "anthropic/claude-sonnet-4"
    resp = await _post(
        app, _req("openai/gpt-4o-mini"), "writer-secret", "writer",
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    # The response carries the failover substitute, and it was NOT 403'd.
    assert data["data"]["model"] == "anthropic/claude-sonnet-4"


# Operator bypasses the pin (manages the fleet).
@pytest.mark.asyncio
async def test_operator_bypasses_pin(pin_setup):
    app = pin_setup["app"]
    resp = await _post(
        app, _req("anthropic/claude-opus-4"), "operator-secret", "operator",
    )
    assert resp.status_code == 200, resp.text


# allowed_models widens the pin without changing the configured model.
@pytest.mark.asyncio
async def test_allowed_models_widens_pin(pin_setup):
    app = pin_setup["app"]
    pin_setup["cfg_holder"]["cfg"]["agents"]["writer"]["allowed_models"] = [
        "anthropic/claude-opus-4",
    ]
    # Configured model still works.
    resp = await _post(
        app, _req("openai/gpt-4o-mini"), "writer-secret", "writer",
    )
    assert resp.status_code == 200, resp.text
    # The extra allowed model also works.
    resp = await _post(
        app, _req("anthropic/claude-opus-4"), "writer-secret", "writer",
    )
    assert resp.status_code == 200, resp.text
    # An un-listed model is still rejected.
    resp = await _post(
        app, _req("openai/gpt-4-turbo"), "writer-secret", "writer",
    )
    assert resp.status_code == 403


# Incompatible requested model rejected at the proxy boundary (interim
# is_model_compatible gate) even when it IS the configured/pinned model.
@pytest.mark.asyncio
async def test_incompatible_requested_model_rejected(pin_setup):
    app = pin_setup["app"]
    pin_setup["vault"]._compatible = False
    resp = await _post(
        app, _req("openai/gpt-4o-mini"), "writer-secret", "writer",
    )
    assert resp.status_code == 403, resp.text
    assert "not compatible" in resp.text


def _embed_req(model: str) -> dict:
    return {
        "service": "llm",
        "action": "embed",
        "params": {
            "model": model,
            "input": "some text to embed",
        },
        "timeout": 30,
    }


# Embedding calls use a fixed embedding model distinct from the agent's chat
# model, so the chat-model pin MUST NOT 403 them — otherwise every memory
# write and vector search breaks. The pin is gated on action != "embed".
@pytest.mark.asyncio
async def test_embed_off_allowlist_model_not_blocked(pin_setup):
    app = pin_setup["app"]
    # text-embedding-3-small is NOT in writer's allowed set (writer is pinned
    # to openai/gpt-4o-mini) — but as an embed action it must pass the pin and
    # reach dispatch, not 403.
    resp = await _post(
        app, _embed_req("text-embedding-3-small"), "writer-secret", "writer",
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True
    # It reached the vault (the pin did not short-circuit it).
    assert pin_setup["vault"].calls[-1]["model"] == "text-embedding-3-small"


# The pin still works for chat with that SAME off-allowlist model — embed is
# the only exempt action, chat/streaming-chat stay pinned.
@pytest.mark.asyncio
async def test_chat_same_off_allowlist_model_still_403(pin_setup):
    app = pin_setup["app"]
    resp = await _post(
        app, _req("text-embedding-3-small"), "writer-secret", "writer",
    )
    assert resp.status_code == 403, resp.text
    assert "not authorized to use model" in resp.text


# REGRESSION (fail-open): an agent with NO explicit ``model`` field in its
# config is NOT pinned — the old implementation fell back to the global
# ``llm.default_model`` so the allowed set was never empty, which pinned
# (and 403'd) operator-created agents whose config carried no model row.
# Now the pin fails open: any requested model passes the pin and reaches
# dispatch.
@pytest.mark.asyncio
async def test_agent_without_explicit_model_not_pinned(pin_setup):
    app = pin_setup["app"]
    # Drop writer's explicit model — config has no per-agent model. The
    # global default_model is still openai/gpt-4o-mini, but it must NOT be
    # used as a pin source anymore.
    pin_setup["cfg_holder"]["cfg"]["agents"]["writer"].pop("model", None)
    # An arbitrary model unrelated to the global default passes the pin.
    resp = await _post(
        app, _req("anthropic/claude-opus-4"), "writer-secret", "writer",
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True
    # It reached the vault (the pin did not short-circuit it).
    assert pin_setup["vault"].calls[-1]["model"] == "anthropic/claude-opus-4"


# An agent with no config row at all (not present under ``agents``) is also
# unpinned — same fail-open path.
@pytest.mark.asyncio
async def test_agent_missing_from_config_not_pinned(pin_setup):
    app = pin_setup["app"]
    pin_setup["cfg_holder"]["cfg"]["agents"].pop("writer", None)
    resp = await _post(
        app, _req("anthropic/claude-opus-4"), "writer-secret", "writer",
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True


# Provider-prefix-insensitive compare: an explicit config of
# ``anthropic/claude-...`` must accept a request for the bare
# ``claude-...`` (the prefix alone must not false-trip the pin).
@pytest.mark.asyncio
async def test_prefix_insensitive_configured_prefixed_request_bare(pin_setup):
    app = pin_setup["app"]
    pin_setup["cfg_holder"]["cfg"]["agents"]["writer"]["model"] = (
        "anthropic/claude-3-5-sonnet"
    )
    resp = await _post(
        app, _req("claude-3-5-sonnet"), "writer-secret", "writer",
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True
    assert pin_setup["vault"].calls[-1]["model"] == "claude-3-5-sonnet"


# ...and the reverse: configured bare, request prefixed.
@pytest.mark.asyncio
async def test_prefix_insensitive_configured_bare_request_prefixed(pin_setup):
    app = pin_setup["app"]
    pin_setup["cfg_holder"]["cfg"]["agents"]["writer"]["model"] = (
        "claude-3-5-sonnet"
    )
    resp = await _post(
        app, _req("anthropic/claude-3-5-sonnet"), "writer-secret", "writer",
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True
    assert (
        pin_setup["vault"].calls[-1]["model"] == "anthropic/claude-3-5-sonnet"
    )


# Prefix-insensitivity must not weaken the pin: a genuinely different bare
# name (same provider prefix) is still 403'd.
@pytest.mark.asyncio
async def test_prefix_insensitive_does_not_allow_different_bare_name(pin_setup):
    app = pin_setup["app"]
    pin_setup["cfg_holder"]["cfg"]["agents"]["writer"]["model"] = (
        "anthropic/claude-3-5-sonnet"
    )
    resp = await _post(
        app, _req("anthropic/claude-opus-4"), "writer-secret", "writer",
    )
    assert resp.status_code == 403, resp.text
    assert "not authorized to use model" in resp.text
