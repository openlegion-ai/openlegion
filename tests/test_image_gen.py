"""Tests for image generation: tool, credential handlers, and cost tracking."""

from __future__ import annotations

import base64
import os
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.host.costs import CostTracker
from src.host.credentials import CredentialVault
from src.shared.types import APIProxyRequest

# ── Tool tests (mock mesh_client) ─────────────────────────────


@pytest.fixture
def tmp_artifacts(tmp_path):
    """Patch _ARTIFACTS_DIR to a temp directory."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    with patch("src.agent.builtins.image_gen_tool._ARTIFACTS_DIR", artifacts):
        yield artifacts


@pytest.fixture
def mock_mesh_client():
    client = AsyncMock()
    client.agent_id = "test-agent"
    client.project_name = "test-project"
    return client


def _make_success_response(image_b64="aW1hZ2VkYXRh", mime_type="image/png"):
    return {
        "success": True,
        "data": {
            "image_base64": image_b64,
            "mime_type": mime_type,
            "model": "gemini-2.0-flash-image-generation",
            "fixed_cost_usd": 0.04,
        },
    }


async def test_generate_image_success(mock_mesh_client, tmp_artifacts):
    from src.agent.builtins.image_gen_tool import generate_image

    mock_mesh_client.image_generate = AsyncMock(return_value=_make_success_response())
    mock_mesh_client.write_blackboard = AsyncMock(return_value={})

    result = await generate_image(
        "a cat wearing a hat",
        mesh_client=mock_mesh_client,
        workspace_manager=MagicMock(),
    )

    assert result["status"] == "image generated"
    assert "path" in result
    assert result["provider"]  # non-empty (model name or provider)
    assert "_image" in result
    assert result["_image"]["data"] == "aW1hZ2VkYXRh"
    assert result["_image"]["media_type"] == "image/png"
    # Verify file was saved
    saved = tmp_artifacts / "a_cat_wearing_a_hat.png"
    assert saved.exists()
    assert saved.read_bytes() == base64.b64decode("aW1hZ2VkYXRh")


async def test_generate_image_no_mesh_client(tmp_artifacts):
    from src.agent.builtins.image_gen_tool import generate_image

    result = await generate_image("test prompt", mesh_client=None)
    assert "error" in result
    assert "mesh connectivity" in result["error"]


async def test_generate_image_empty_prompt(mock_mesh_client, tmp_artifacts):
    from src.agent.builtins.image_gen_tool import generate_image

    result = await generate_image("", mesh_client=mock_mesh_client)
    assert "error" in result
    assert "prompt is required" in result["error"]

    result2 = await generate_image("   ", mesh_client=mock_mesh_client)
    assert "error" in result2


async def test_generate_image_invalid_size(mock_mesh_client, tmp_artifacts):
    from src.agent.builtins.image_gen_tool import generate_image

    mock_mesh_client.image_generate = AsyncMock(return_value=_make_success_response())
    mock_mesh_client.write_blackboard = AsyncMock(return_value={})

    result = await generate_image(
        "test",
        size="huge",
        mesh_client=mock_mesh_client,
        workspace_manager=MagicMock(),
    )

    # Should fall back to square, not error
    assert result["status"] == "image generated"
    mock_mesh_client.image_generate.assert_called_once()
    call_kwargs = mock_mesh_client.image_generate.call_args
    assert call_kwargs.kwargs.get("size", call_kwargs[1].get("size")) == "square"


async def test_generate_image_custom_filename(mock_mesh_client, tmp_artifacts):
    from src.agent.builtins.image_gen_tool import generate_image

    mock_mesh_client.image_generate = AsyncMock(return_value=_make_success_response())
    mock_mesh_client.write_blackboard = AsyncMock(return_value={})

    result = await generate_image(
        "test",
        filename="my_image.png",
        mesh_client=mock_mesh_client,
        workspace_manager=MagicMock(),
    )

    assert result["status"] == "image generated"
    assert "my_image.png" in result["path"]
    assert (tmp_artifacts / "my_image.png").exists()


async def test_generate_image_api_error(mock_mesh_client, tmp_artifacts):
    from src.agent.builtins.image_gen_tool import generate_image

    mock_mesh_client.image_generate = AsyncMock(
        return_value={"success": False, "error": "Rate limited"},
    )

    result = await generate_image("test", mesh_client=mock_mesh_client)
    assert "error" in result
    assert "Rate limited" in result["error"]


async def test_generate_image_exception(mock_mesh_client, tmp_artifacts):
    from src.agent.builtins.image_gen_tool import generate_image

    mock_mesh_client.image_generate = AsyncMock(side_effect=httpx.ConnectError("timeout"))

    result = await generate_image("test", mesh_client=mock_mesh_client)
    assert "error" in result
    assert "failed" in result["error"].lower()


async def test_generate_image_path_traversal(mock_mesh_client, tmp_artifacts):
    from src.agent.builtins.image_gen_tool import generate_image

    mock_mesh_client.image_generate = AsyncMock(return_value=_make_success_response())
    mock_mesh_client.write_blackboard = AsyncMock(return_value={})

    # Directory separators are stripped by Path.name, unsafe chars replaced
    result = await generate_image(
        "test",
        filename="../../etc/passwd",
        mesh_client=mock_mesh_client,
        workspace_manager=MagicMock(),
    )
    # Should succeed but with sanitized filename, not traverse
    assert result.get("status") == "image generated"
    assert "etc" not in result.get("path", "")
    assert ".." not in result.get("path", "")


async def test_generate_image_filename_no_extension(mock_mesh_client, tmp_artifacts):
    from src.agent.builtins.image_gen_tool import generate_image

    mock_mesh_client.image_generate = AsyncMock(return_value=_make_success_response())
    mock_mesh_client.write_blackboard = AsyncMock(return_value={})

    result = await generate_image(
        "test",
        filename="myfile",
        mesh_client=mock_mesh_client,
        workspace_manager=MagicMock(),
    )
    assert result["status"] == "image generated"
    assert result["path"].endswith(".png")


async def test_generate_image_empty_slug_prompt(mock_mesh_client, tmp_artifacts):
    from src.agent.builtins.image_gen_tool import generate_image

    mock_mesh_client.image_generate = AsyncMock(return_value=_make_success_response())
    mock_mesh_client.write_blackboard = AsyncMock(return_value={})

    # Non-alphanumeric prompt produces empty slug
    result = await generate_image(
        "!!!???",
        mesh_client=mock_mesh_client,
        workspace_manager=MagicMock(),
    )
    assert result["status"] == "image generated"
    assert "generated_image.png" in result["path"]


async def test_generate_image_no_project(tmp_artifacts):
    """Non-project agent should skip blackboard write without error."""
    from src.agent.builtins.image_gen_tool import generate_image

    client = AsyncMock()
    client.agent_id = "standalone-agent"
    client.project_name = None  # No project
    client.image_generate = AsyncMock(return_value=_make_success_response())

    result = await generate_image(
        "test",
        mesh_client=client,
        workspace_manager=MagicMock(),
    )
    assert result["status"] == "image generated"
    # write_blackboard should not have been called
    client.write_blackboard.assert_not_called()


# ── Handler tests (mock httpx) ────────────────────────────────


@pytest.fixture
def vault_with_gemini(monkeypatch):
    monkeypatch.setenv("OPENLEGION_SYSTEM_GEMINI_API_KEY", "test-gemini-key")
    v = CredentialVault()
    return v


@pytest.fixture
def vault_with_openai(monkeypatch):
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "test-openai-key")
    v = CredentialVault()
    return v


def _gemini_response_json():
    return {
        "candidates": [{
            "content": {
                "parts": [{
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": "Z2VtaW5pX2ltYWdl",
                    },
                }],
            },
        }],
    }


def _openai_response_json():
    return {
        "data": [{
            "b64_json": "b3BlbmFpX2ltYWdl",
        }],
    }


async def test_handle_gemini_success(vault_with_gemini):
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = _gemini_response_json()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.is_closed = False
    vault_with_gemini._http_client = mock_client

    req = APIProxyRequest(
        service="image_gen", action="generate",
        params={"prompt": "a dog", "provider": "gemini"},
    )
    result = await vault_with_gemini._handle_image_gen(req)
    assert result.success
    assert result.data["image_base64"] == "Z2VtaW5pX2ltYWdl"
    assert result.data["mime_type"] == "image/png"
    assert result.data["fixed_cost_usd"] > 0


async def test_handle_gemini_no_key():
    v = CredentialVault()
    req = APIProxyRequest(
        service="image_gen", action="generate",
        params={"prompt": "a dog", "provider": "gemini"},
    )
    result = await v._handle_image_gen(req)
    assert not result.success
    assert "not configured" in result.error


async def test_handle_openai_success(vault_with_openai):
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = _openai_response_json()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.is_closed = False
    vault_with_openai._http_client = mock_client

    req = APIProxyRequest(
        service="image_gen", action="generate",
        params={"prompt": "a dog", "provider": "openai", "size": "landscape"},
    )
    result = await vault_with_openai._handle_image_gen(req)
    assert result.success
    assert result.data["image_base64"] == "b3BlbmFpX2ltYWdl"
    assert result.data["mime_type"] == "image/png"
    assert result.data["model"] == "dall-e-3"
    assert result.data["fixed_cost_usd"] > 0

    # Verify DALL-E size mapping
    call_body = mock_client.post.call_args.kwargs.get("json", {})
    assert call_body["size"] == "1792x1024"


async def test_handle_openai_no_key():
    v = CredentialVault()
    req = APIProxyRequest(
        service="image_gen", action="generate",
        params={"prompt": "a dog", "provider": "openai"},
    )
    result = await v._handle_image_gen(req)
    assert not result.success
    assert "not configured" in result.error


async def test_handle_unknown_provider():
    v = CredentialVault()
    req = APIProxyRequest(
        service="image_gen", action="generate",
        params={"prompt": "a dog", "provider": "midjourney"},
    )
    result = await v._handle_image_gen(req)
    assert not result.success
    assert "Unknown image_gen provider" in result.error


async def test_handle_empty_prompt():
    v = CredentialVault()
    req = APIProxyRequest(
        service="image_gen", action="generate",
        params={"prompt": "", "provider": "gemini"},
    )
    result = await v._handle_image_gen(req)
    assert not result.success
    assert "prompt is required" in result.error


async def test_gemini_no_image_in_response(vault_with_gemini):
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.is_closed = False
    vault_with_gemini._http_client = mock_client

    req = APIProxyRequest(
        service="image_gen", action="generate",
        params={"prompt": "a dog", "provider": "gemini"},
    )
    result = await vault_with_gemini._handle_image_gen(req)
    assert not result.success
    assert "no image data" in result.error.lower()


async def test_openai_size_mapping():
    v = CredentialVault()
    assert v._OPENAI_SIZE_MAP["square"] == "1024x1024"
    assert v._OPENAI_SIZE_MAP["landscape"] == "1792x1024"
    assert v._OPENAI_SIZE_MAP["portrait"] == "1024x1792"


async def test_gemini_model_fallback(vault_with_gemini):
    """When first model returns 403, falls back to next model in chain."""
    mock_403 = MagicMock()
    mock_403.status_code = 403
    mock_403.is_success = False
    mock_403.text = "Model not found"

    mock_ok = MagicMock()
    mock_ok.status_code = 200
    mock_ok.is_success = True
    mock_ok.json.return_value = _gemini_response_json()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[mock_403, mock_ok])
    mock_client.is_closed = False
    vault_with_gemini._http_client = mock_client

    req = APIProxyRequest(
        service="image_gen", action="generate",
        params={"prompt": "a dog", "provider": "gemini"},
    )
    result = await vault_with_gemini._handle_image_gen(req)
    assert result.success
    assert result.data["image_base64"] == "Z2VtaW5pX2ltYWdl"
    # Should have been called twice (first model 403, second succeeds)
    assert mock_client.post.call_count == 2


async def test_gemini_all_models_unavailable(vault_with_gemini):
    """When all models return 403, returns clear error."""
    mock_403 = MagicMock()
    mock_403.status_code = 403
    mock_403.is_success = False
    mock_403.text = "Model not found"

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_403)
    mock_client.is_closed = False
    vault_with_gemini._http_client = mock_client

    req = APIProxyRequest(
        service="image_gen", action="generate",
        params={"prompt": "a dog", "provider": "gemini"},
    )
    result = await vault_with_gemini._handle_image_gen(req)
    assert not result.success
    assert "unavailable" in result.error.lower()


# ── Cost tracking tests ───────────────────────────────────────


class TestFixedCostTracking:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self._tmpdir, "costs.db")
        self.tracker = CostTracker(db_path=self.db_path)

    def teardown_method(self):
        self.tracker.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_track_fixed_cost(self):
        result = self.tracker.track_fixed_cost("agent1", "dall-e-3", 0.04)
        assert result["cost"] == 0.04
        assert result["over_budget"] is False

        # Verify it recorded in the DB
        spend = self.tracker.get_spend("agent1", "today")
        assert spend["total_cost"] == 0.04
        assert spend["total_tokens"] == 0  # Fixed cost has no tokens

    def test_fixed_cost_budget_enforcement(self):
        self.tracker.set_budget("agent1", daily_usd=0.05, monthly_usd=1.0)
        self.tracker.track_fixed_cost("agent1", "dall-e-3", 0.04)
        result = self.tracker.track_fixed_cost("agent1", "dall-e-3", 0.04)
        assert result["over_budget"] is True

    def test_fixed_cost_in_spend_report(self):
        self.tracker.track_fixed_cost("agent1", "gemini-image", 0.04)
        self.tracker.track("agent1", "openai/gpt-4o", 10000, 5000)

        spend = self.tracker.get_spend("agent1", "today")
        assert spend["total_cost"] > 0.04  # Image cost + LLM cost
        assert "gemini-image" in spend["by_model"]
        assert spend["by_model"]["gemini-image"]["cost"] == 0.04
        assert spend["by_model"]["gemini-image"]["total"] == 0  # zero tokens


# ── Integration: execute_api_call ─────────────────────────────


async def test_execute_api_call_image_gen_tracks_fixed_cost(monkeypatch):
    """Fixed cost from image_gen response is tracked via track_fixed_cost."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_GEMINI_API_KEY", "test-key")
    v = CredentialVault()

    tmpdir = tempfile.mkdtemp()
    tracker = CostTracker(db_path=os.path.join(tmpdir, "costs.db"))
    v.cost_tracker = tracker
    tracker.set_budget("agent1", daily_usd=10.0, monthly_usd=100.0)

    # Mock the Gemini HTTP call
    mock_resp = MagicMock()
    mock_resp.is_success = True
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "image/png", "data": "abc123"}},
        ]}}],
    }
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.is_closed = False
    v._http_client = mock_client

    req = APIProxyRequest(
        service="image_gen", action="generate",
        params={"prompt": "a cat", "provider": "gemini"},
    )
    result = await v.execute_api_call(req, agent_id="agent1")
    assert result.success

    # Verify fixed cost was tracked
    spend = tracker.get_spend("agent1", "today")
    assert spend["total_cost"] > 0
    assert spend["total_tokens"] == 0  # No token usage for image gen

    tracker.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


async def test_execute_api_call_image_gen_budget_blocked(monkeypatch):
    """Image gen is blocked when budget is exhausted."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_GEMINI_API_KEY", "test-key")
    v = CredentialVault()

    tmpdir = tempfile.mkdtemp()
    tracker = CostTracker(db_path=os.path.join(tmpdir, "costs.db"))
    v.cost_tracker = tracker
    tracker.set_budget("agent1", daily_usd=0.01, monthly_usd=100.0)
    # Exhaust budget
    tracker.track_fixed_cost("agent1", "prior", 0.02)

    req = APIProxyRequest(
        service="image_gen", action="generate",
        params={"prompt": "a cat", "provider": "gemini"},
    )
    result = await v.execute_api_call(req, agent_id="agent1")
    assert not result.success
    assert "Budget exceeded" in result.error

    tracker.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


# ── Handler registration test ─────────────────────────────────


def test_image_gen_handler_registered():
    v = CredentialVault()
    assert "image_gen" in v.service_handlers
