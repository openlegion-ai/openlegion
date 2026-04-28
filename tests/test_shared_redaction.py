"""Tests for the unified :mod:`src.shared.redaction` module (Phase 1.3).

Covers three layers:

1. Unit tests for ``redact_string`` / ``redact_url`` / ``deep_redact`` — each
   behavioral rule has at least one positive and one negative assertion.
2. A **corpus** test that reads ``tests/fixtures/redaction_corpus.json`` —
   the gate against missing a known leak pattern. Each fixture entry lists
   tokens that must be redacted and tokens that must be preserved.
3. Backward-compat checks for the two shim modules (``src/browser/redaction``
   and the agent-side browser tool) — ensures old imports still work.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from src.shared.redaction import (
    _REDACT_PATTERNS,
    SECRET_PATTERNS,
    SENSITIVE_QUERY_PARAMS,
    deep_redact,
    redact_string,
    redact_url,
)

CORPUS_PATH = Path(__file__).parent / "fixtures" / "redaction_corpus.json"
_REDACTED = "[REDACTED]"


# ── Unit tests: redact_string ───────────────────────────────────────────────


class TestRedactString:
    @pytest.mark.parametrize("secret", [
        "sk-abcdefghijklmnopqrstuvwxyz0123456789",
        "sk-ant-api01-" + "A" * 30,
        "gho_" + "a" * 40,
        "github_pat_11ABCDEF" + "0" * 30,
        "xoxb-" + "1" * 40,
        "xoxp-" + "2" * 40,
        "AKIAIOSFODNN7EXAMPLE",
    ])
    def test_provider_patterns_redacted(self, secret):
        assert redact_string(f"key={secret}") == f"key={_REDACTED}"

    def test_long_hex_blob_redacted(self):
        out = redact_string("hash:" + "ab" * 25)
        assert _REDACTED in out

    def test_long_base64_blob_redacted(self):
        blob = "A" * 60 + "=="
        assert _REDACTED in redact_string(f"token:{blob}")

    def test_short_hex_preserved(self):
        # Commit hashes, short refs — below the 40-char threshold.
        assert redact_string("commit abc1234 deadbee") == "commit abc1234 deadbee"

    def test_empty_and_none_preserved(self):
        assert redact_string("") == ""
        assert redact_string(None) is None

    def test_jwt_redacted_anywhere_in_string(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signaturepartatleasttenchars"
        assert redact_string(f"Bearer {jwt}") == f"Bearer {_REDACTED}"


# ── Unit tests: redact_url ──────────────────────────────────────────────────


class TestRedactUrl:
    def test_strips_userinfo(self):
        out = redact_url("https://alice:pw@host.example/admin")
        assert "alice" not in out
        assert "pw" not in out
        assert "host.example/admin" in out

    def test_drops_fragment(self):
        out = redact_url("https://app.example/cb?x=1#access_token=abc.def.ghi")
        assert "access_token" not in out
        assert "#" not in out
        assert "x=1" in out  # non-sensitive query preserved

    def test_redacts_sensitive_query_keys_keeps_keys(self):
        out = redact_url("https://svc.example/?api_key=mysecret&public=hello")
        assert "api_key=" in out  # key preserved for debug
        assert "mysecret" not in out
        assert _REDACTED in out
        assert "public=hello" in out

    def test_case_insensitive_query_key_match(self):
        out = redact_url("https://svc.example/?API_KEY=mysecret")
        assert "mysecret" not in out
        assert _REDACTED in out

    def test_oauth_code_and_state_stripped(self):
        out = redact_url("https://app.example/cb?code=authz123&state=csrf456")
        assert "authz123" not in out
        assert "csrf456" not in out

    def test_aws_sigv4_params_stripped(self):
        sig = "deadbeefcafebabe" * 3
        out = redact_url(f"https://bucket.s3.amazonaws.com/x?X-Amz-Signature={sig}")
        assert sig not in out
        assert "X-Amz-Signature" in out

    def test_jwt_path_segment_redacted(self):
        jwt_seg = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhYmNkIn0.signaturemustbetenchars"
        out = redact_url(f"https://email.example/u/{jwt_seg}/click")
        assert jwt_seg not in out
        assert _REDACTED in out
        assert "/click" in out

    def test_plain_url_untouched(self):
        url = "https://example.com/docs/page.html?topic=security"
        assert redact_url(url) == url

    def test_non_url_falls_through_to_string(self):
        # String without "://" takes the pattern-only path.
        out = redact_url("my key: sk-abcdefghijklmnopqrstuvwxyz0123456789")
        assert _REDACTED in out
        assert "my key:" in out

    def test_idempotent_on_already_redacted(self):
        once = redact_url("https://svc.example/?api_key=abc123xyz")
        twice = redact_url(once)
        assert once == twice

    @pytest.mark.parametrize("scheme", ["javascript", "data", "blob", "vbscript", "file"])
    def test_non_web_schemes_skip_structural_parsing(self, scheme):
        """Exotic URL schemes don't follow authority/path/query/fragment —
        structural parsing would produce nonsense. Fall through to the
        pattern sweep only."""
        probe = f"{scheme}://host/not-a-real-path?api_key=visible"
        out = redact_url(probe)
        # The scheme is preserved (we don't parse-and-reassemble).
        assert out.startswith(f"{scheme}://")
        # Pattern-level sweep still runs — no embedded regex-matching
        # secret in this test string, so nothing should change other
        # than it being returned as-is.
        assert out == probe

    def test_api_token_variants_stripped(self):
        """Common API-token param names beyond plain ``token`` are covered."""
        for key in ("api_token", "apitoken", "api-token", "bearer_token"):
            out = redact_url(f"https://svc.example/?{key}=zzz-secret-zzz")
            assert "zzz-secret-zzz" not in out, key
            assert _REDACTED in out

    def test_relative_url_with_query_string_redacted(self):
        """OAuth callback paths and copy-pasted URL fragments often arrive
        without a scheme (``/cb?code=...&state=...``). Structural query
        stripping must still kick in."""
        out = redact_url("/cb?code=authgrant123&state=csrfval&keep=yes")
        assert "authgrant123" not in out
        assert "csrfval" not in out
        assert "keep=yes" in out
        assert "/cb?" in out

    def test_plain_text_without_query_uses_patterns_only(self):
        """A bare relative path (no ``?``) should not be parsed as a URL."""
        out = redact_url("/posts/42")
        assert out == "/posts/42"


class TestSensitiveQueryParams:
    def test_common_auth_params_present(self):
        for expected in (
            "api_key", "apikey", "api-key",
            "api_token", "apitoken", "api-token",
            "token", "bearer_token",
            "password", "secret",
        ):
            assert expected in SENSITIVE_QUERY_PARAMS, expected

    def test_oauth_params_present(self):
        for expected in ("code", "state", "access_token", "refresh_token", "id_token"):
            assert expected in SENSITIVE_QUERY_PARAMS, expected

    def test_aws_gcs_azure_params_present(self):
        for expected in ("x-amz-signature", "x-goog-signature", "sig"):
            assert expected in SENSITIVE_QUERY_PARAMS, expected


# ── Operator allowlist opt-out ──────────────────────────────────────────────


class TestOperatorAllowlist:
    def test_default_is_empty(self):
        """Empty env var → strict deny-by-default."""
        with mock.patch.dict(os.environ, {"OPENLEGION_REDACTION_URL_QUERY_ALLOW": ""}):
            out = redact_url("https://svc/?api_key=xyz")
            assert "xyz" not in out

    def test_explicit_allowlist_preserves_value(self):
        """Operator opts api_key out of redaction (e.g. verified non-sensitive)."""
        with mock.patch.dict(os.environ, {"OPENLEGION_REDACTION_URL_QUERY_ALLOW": "api_key"}):
            out = redact_url("https://svc/?api_key=notasecret")
            assert "notasecret" in out

    def test_allowlist_case_insensitive(self):
        with mock.patch.dict(os.environ, {"OPENLEGION_REDACTION_URL_QUERY_ALLOW": "API_KEY"}):
            out = redact_url("https://svc/?api_key=notasecret")
            assert "notasecret" in out

    def test_allowlist_does_not_affect_other_params(self):
        """Allowing one key doesn't relax the rest."""
        with mock.patch.dict(os.environ, {"OPENLEGION_REDACTION_URL_QUERY_ALLOW": "api_key"}):
            out = redact_url("https://svc/?api_key=ok&password=nope")
            assert "ok" in out
            assert "nope" not in out


# ── Unit tests: deep_redact ─────────────────────────────────────────────────


class TestDeepRedact:
    def test_nested_dict(self):
        obj = {
            "outer": {
                "token": "sk-abcdefghijklmnopqrstuvwxyz1234567890",
                "note": "safe",
            },
            "list": ["ok", "sk-anothersecretthatsverylonganymisalaiki00"],
        }
        red = deep_redact(obj)
        assert red["outer"]["token"] == _REDACTED
        assert red["outer"]["note"] == "safe"
        assert red["list"][0] == "ok"
        assert _REDACTED in red["list"][1]

    def test_url_inside_dict_gets_url_redaction(self):
        obj = {"callback": "https://app/cb?api_key=zzz"}
        red = deep_redact(obj)
        assert "zzz" not in red["callback"]
        assert "https://app/cb?" in red["callback"]

    def test_primitive_types_pass_through(self):
        assert deep_redact(42) == 42
        assert deep_redact(True) is True
        assert deep_redact(None) is None
        assert deep_redact(3.14) == 3.14

    def test_tuple_returned_as_tuple(self):
        assert deep_redact(("a", "sk-abcdefghijklmnopqrstuvwxyz1234567890")) == (
            "a", _REDACTED,
        )


# ── Corpus gate ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def corpus():
    """Load the redaction corpus fixture once per test module."""
    with CORPUS_PATH.open() as f:
        return json.load(f)


class TestCorpusGate:
    def test_corpus_file_exists_and_is_nonempty(self, corpus):
        assert isinstance(corpus, list)
        assert len(corpus) >= 25, "Corpus thinner than expected — did someone trim?"

    def test_every_redacted_token_stripped(self, corpus):
        """Each entry names tokens that MUST disappear from the redacted output."""
        for entry in corpus:
            out = redact_url(entry["input"])
            for tok in entry.get("expect_redacted", []):
                assert tok not in out, (
                    f"[{entry['name']}] leaked token {tok!r} — redacted output: {out!r}"
                )

    def test_every_preserved_token_survives(self, corpus):
        """Each entry names tokens that MUST remain visible — guards against
        over-redaction that would ruin debuggability."""
        for entry in corpus:
            out = redact_url(entry["input"])
            for tok in entry.get("expect_preserved", []):
                # Allow for trailing-slash / fragment-drop normalizations —
                # substring match is sufficient.
                assert tok in out, (
                    f"[{entry['name']}] lost preserved token {tok!r} — output: {out!r}"
                )


# ── Backward-compat shims ───────────────────────────────────────────────────


class TestBrowserRedactorShim:
    """Existing call sites in ``src/browser/service.py`` use
    :class:`CredentialRedactor`. Shim must keep passing."""

    def test_redact_delegates(self):
        from src.browser.redaction import CredentialRedactor
        r = CredentialRedactor()
        assert r.redact("agent-1", "key=sk-abcdefghijklmnopqrstuvwxyz1234567890") == (
            f"key={_REDACTED}"
        )

    def test_redact_is_url_aware(self):
        from src.browser.redaction import CredentialRedactor
        r = CredentialRedactor()

        out = r.redact(
            "agent-1",
            "https://svc.example/?api_key=mysecret&public=ok",
        )

        assert "mysecret" not in out
        assert "api_key=" in out
        assert "public=ok" in out
        assert _REDACTED in out

    def test_deep_redact_delegates(self):
        from src.browser.redaction import CredentialRedactor
        r = CredentialRedactor()
        out = r.deep_redact("agent-1", {"x": "sk-abcdefghijklmnopqrstuvwxyz1234567890"})
        assert out["x"] == _REDACTED

    def test_legacy_patterns_constant_still_exported(self):
        from src.browser.redaction import _REDACT_PATTERNS as browser_patterns
        # Same object as the shared module's — no drift possible.
        assert browser_patterns is _REDACT_PATTERNS
        assert browser_patterns is SECRET_PATTERNS


class TestAgentBrowserToolShim:
    """The agent-side ``_redact_credentials`` / ``_deep_redact`` private
    helpers used to live in ``src/agent/builtins/browser_tool.py``. They're
    now thin imports from the shared module; the names must still resolve."""

    def test_agent_side_helpers_still_import(self):
        from src.agent.builtins.browser_tool import (
            _deep_redact,
            _redact_credentials,
        )
        assert callable(_redact_credentials)
        assert callable(_deep_redact)
        assert _redact_credentials("abc") == "abc"
        assert _deep_redact({"a": "b"}) == {"a": "b"}
