"""Unified credential + URL redaction (Phase 1.3).

Single source of truth for stripping secrets from any text that might reach
LLM context, logs, or agent-visible responses. Previously duplicated across
:mod:`src.browser.redaction` and the agent-side ``browser_tool`` module;
both now re-export this module.

Public surface:

- :data:`SECRET_PATTERNS` — module-level list of compiled regexes matching
  common API-key / token shapes (OpenAI, Anthropic, GitHub, Slack, AWS, long
  hex/base64 blobs). Prefix-underscore per CLAUDE.md constant convention.
- :data:`SENSITIVE_QUERY_PARAMS` — frozenset of query-parameter names that
  always have their values stripped from URLs, regardless of pattern match.
  Case-insensitive matching.
- :func:`redact_string(text)` — apply :data:`SECRET_PATTERNS` to a plain
  string.  Backward-compatible with the v1 :class:`CredentialRedactor.redact`
  behavior.
- :func:`redact_url(url)` — component-aware URL redaction. Strips userinfo,
  drops fragments, replaces sensitive-query-param VALUES with ``[REDACTED]``
  (keeping the keys for debuggability), detects JWT-shaped tokens in path
  segments.  Always safe to call on non-URL strings (returns unchanged).
- :func:`deep_redact(obj)` — recursive dict/list/string redaction. Strings
  flow through :func:`redact_string`; strings that look like URLs additionally
  flow through :func:`redact_url`.

Operator controls (env-var-driven):

- ``OPENLEGION_REDACTION_URL_QUERY_ALLOW`` — comma-separated query-param
  names that should NOT be redacted even if they appear sensitive. Intended
  for specific integrations where the operator has verified the value is
  non-sensitive. Empty (default) = strict deny-by-default.

Threat model boundary (out of scope here):

- This module does not detect or redact DOM-level secrets embedded in
  rendered HTML (e.g. a session cookie mirrored into page text). That's the
  browser-service snapshot pipeline's job; it calls :func:`redact_string` on
  the rendered text before returning to agents.
- This module does not redact response bodies of captured network requests.
  Phase 9.1 network-inspection plan forbids body capture entirely; URLs
  alone flow through :func:`redact_url`.
"""

from __future__ import annotations

import os
import re
from urllib.parse import SplitResult, parse_qsl, quote_plus, urlsplit, urlunsplit

# ── Pattern-based secret redaction ──────────────────────────────────────────


SECRET_PATTERNS: list[re.Pattern[str]] = [
    # Provider-specific tokens (prefix-anchored; low false-positive rate)
    re.compile(r"sk-[A-Za-z0-9]{20,}"),              # OpenAI / Anthropic short form
    re.compile(r"sk-ant-api[A-Za-z0-9\-]{20,}"),     # Anthropic
    re.compile(r"gho_[A-Za-z0-9]{36,}"),              # GitHub OAuth access tokens
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),      # GitHub fine-grained PATs
    re.compile(r"xoxb-[A-Za-z0-9\-]{20,}"),           # Slack bot tokens
    re.compile(r"xoxp-[A-Za-z0-9\-]{20,}"),           # Slack user tokens
    re.compile(r"AKIA[A-Z0-9]{16}"),                   # AWS access key IDs
    # Generic long-hex / long-base64 blobs with boundary lookarounds so we
    # don't redact mid-word hex sequences.
    re.compile(r"(?<![A-Za-z0-9])[A-Fa-f0-9]{40,}(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9/+=])"),
    # JWT (three base64-url segments separated by dots, ≥10 chars each).
    # Matches bearer JWTs in logs, auth headers, signed URLs.
    re.compile(r"(?<![A-Za-z0-9._-])"
               r"[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
               r"(?![A-Za-z0-9._-])"),
]

# Backward-compat alias used by the old ``_REDACT_PATTERNS`` name in
# ``src/browser/redaction.py`` and the agent-side browser_tool.  Existing
# imports keep working.
_REDACT_PATTERNS = SECRET_PATTERNS


# ── URL-aware redaction ─────────────────────────────────────────────────────


# Case-insensitive match on KEY. Values are always stripped. Categorized by
# source for maintainability; the flat frozenset is what gets compared.
#
# Generic auth
_GENERIC = {
    "api_key", "apikey", "api-key",
    "key", "token", "auth", "authorization", "access_token",
    "refresh_token", "id_token", "secret", "client_secret",
    "password", "passwd", "pwd", "assertion",
    "session", "session_id", "sessionid",
    "sig", "signature", "hash", "hmac",
}
# OAuth 2.0 flow params — `code` is the authorization grant; `state` is
# CSRF binding but leaking it lets an attacker correlate sessions.
_OAUTH = {"code", "state"}
# AWS SigV4 presigned URL params
_AWS_SIGV4 = {
    "x-amz-algorithm", "x-amz-credential", "x-amz-date",
    "x-amz-expires", "x-amz-signature", "x-amz-signedheaders",
    "x-amz-security-token",
}
# Google Cloud Storage signed URL params
_GCS_SIGNED = {
    "x-goog-algorithm", "x-goog-credential", "x-goog-date",
    "x-goog-expires", "x-goog-signature", "x-goog-signedheaders",
}
# Azure SAS tokens
_AZURE_SAS = {"sv", "sig", "st", "se", "sp", "sr", "spr"}
# Common Supabase / magic-link / password-reset flavors
_MAGIC_LINK = {
    "token_hash", "magic_token", "reset_token",
    "confirmation_token", "invite_token",
}


SENSITIVE_QUERY_PARAMS: frozenset[str] = frozenset(
    p.lower() for p in (
        *_GENERIC, *_OAUTH, *_AWS_SIGV4, *_GCS_SIGNED, *_AZURE_SAS, *_MAGIC_LINK,
    )
)


# JWT in a path segment — common for password reset / unsubscribe links.
_JWT_IN_PATH = re.compile(
    r"^[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}$"
)


_REDACTED_VALUE = "[REDACTED]"


def _operator_allowlist() -> frozenset[str]:
    """Query-param names the operator has opted out of redaction for.

    Intended for specific integrations where the operator has verified the
    value is non-sensitive in their deployment. Default empty =
    strict deny-by-default.
    """
    raw = os.environ.get("OPENLEGION_REDACTION_URL_QUERY_ALLOW", "").strip()
    if not raw:
        return frozenset()
    return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())


def _should_redact_query(key: str) -> bool:
    """Return True when the query-param ``key`` should have its value stripped."""
    k = key.lower()
    if k in _operator_allowlist():
        return False
    return k in SENSITIVE_QUERY_PARAMS


def redact_url(url: str) -> str:
    """Return ``url`` with secrets stripped from every component.

    Safe to call on strings that don't look like URLs — returns them
    unchanged. Designed to be idempotent: redacting an already-redacted URL
    produces the same string.

    Redaction rules:

    * **Userinfo** — ``user:pass@host`` → ``host`` (whole userinfo dropped;
      the username alone can leak account ownership, so we strip both).
    * **Query params** — values of keys in :data:`SENSITIVE_QUERY_PARAMS`
      are replaced with ``[REDACTED]``. Keys preserved so the shape of the
      URL is readable for debugging.
    * **Fragment** — dropped entirely. Fragments often carry OAuth implicit
      tokens (``#access_token=...``) and rarely carry useful info for logs.
    * **Path segments** — any segment matching the JWT shape gets replaced
      with ``[REDACTED]`` in place.
    * **All components** — after the above, :func:`redact_string` runs one
      more pass so pattern-matching secrets anywhere in the string get
      caught (e.g. an OpenAI key accidentally baked into a path).
    """
    if not url or "://" not in url:
        # Bare domains / non-URL strings: fall through to string patterns only.
        return redact_string(url)

    try:
        parts = urlsplit(url)
    except ValueError:
        # Malformed URL — treat as opaque string.
        return redact_string(url)

    # 1. Netloc: strip userinfo
    netloc = parts.netloc
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[-1]

    # 2. Path: redact JWT-shaped segments
    path = parts.path
    if path:
        segments = path.split("/")
        segments = [
            _REDACTED_VALUE if _JWT_IN_PATH.match(seg) else seg
            for seg in segments
        ]
        path = "/".join(segments)

    # 3. Query: strip sensitive values, keep keys.
    #
    # We rebuild manually instead of calling ``urlencode`` because the
    # default encoder runs both key and value through ``quote_plus``, which
    # would turn our ``[REDACTED]`` sentinel into ``%5BREDACTED%5D`` — still
    # correct semantically, but uglier in logs and breaks simple substring
    # checks that operators use when grep'ing traces.
    query = parts.query
    if query:
        # keep_blank_values=True so we don't lose shape on empty params
        pairs = parse_qsl(query, keep_blank_values=True)
        parts_out: list[str] = []
        for k, v in pairs:
            enc_key = quote_plus(k)
            if _should_redact_query(k):
                # Literal sentinel, no encoding — ``[`` and ``]`` stay visible.
                parts_out.append(f"{enc_key}={_REDACTED_VALUE}")
            else:
                parts_out.append(f"{enc_key}={quote_plus(v)}")
        query = "&".join(parts_out)

    # 4. Fragment: drop entirely
    rebuilt = urlunsplit(SplitResult(parts.scheme, netloc, path, query, ""))

    # 5. Final pattern-sweep in case a token is embedded where we didn't look
    return redact_string(rebuilt)


def redact_string(text: str) -> str:
    """Apply :data:`SECRET_PATTERNS` to ``text`` and return the result."""
    if not text:
        return text
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(_REDACTED_VALUE, text)
    return text


# ── Deep recursion over JSON-shaped structures ──────────────────────────────


def _looks_like_url(s: str) -> bool:
    """Heuristic: bare-minimum check to avoid running urlsplit on plain text.

    Accepts strings with an explicit ``://`` scheme marker. Conservative
    on purpose — false negatives fall back to pattern-only redaction which
    is still safe.
    """
    return "://" in s and len(s) < 4096


def deep_redact(obj):
    """Recursively redact secrets from any JSON-serializable value.

    - Strings flow through :func:`redact_url` if they look like URLs, else
      :func:`redact_string`.
    - Dicts and lists recurse.
    - Tuples are supported but returned as tuples.
    - Other values (int, bool, None, float) pass through unchanged.
    """
    if isinstance(obj, str):
        if _looks_like_url(obj):
            return redact_url(obj)
        return redact_string(obj)
    if isinstance(obj, dict):
        return {k: deep_redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_redact(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(deep_redact(item) for item in obj)
    return obj
