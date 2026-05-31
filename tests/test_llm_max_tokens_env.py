"""Unit tests for set_llm_max_tokens_env — the shared helper that propagates
a per-agent max_output_tokens cap into the container env across every
restart-from-config path (CLI start, dashboard restart, REPL /restart,
fleet-template apply)."""

from __future__ import annotations

from src.shared.utils import set_llm_max_tokens_env


def test_sets_env_for_valid_int():
    env: dict[str, str] = {}
    set_llm_max_tokens_env(env, {"max_output_tokens": 32000})
    assert env["LLM_MAX_TOKENS"] == "32000"


def test_noop_when_absent():
    env: dict[str, str] = {}
    set_llm_max_tokens_env(env, {"model": "anthropic/claude-sonnet-4-6"})
    assert "LLM_MAX_TOKENS" not in env


def test_noop_for_bool():
    # bool is an int subclass — must not become "1"/"0".
    env: dict[str, str] = {}
    set_llm_max_tokens_env(env, {"max_output_tokens": True})
    assert "LLM_MAX_TOKENS" not in env


def test_noop_for_non_int_value():
    env: dict[str, str] = {}
    set_llm_max_tokens_env(env, {"max_output_tokens": "32000"})
    assert "LLM_MAX_TOKENS" not in env


def test_does_not_clobber_other_keys():
    env = {"ALLOWED_TOOLS": "a,b"}
    set_llm_max_tokens_env(env, {"max_output_tokens": 16384})
    assert env["ALLOWED_TOOLS"] == "a,b"
    assert env["LLM_MAX_TOKENS"] == "16384"
