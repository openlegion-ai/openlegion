"""``deep_redact`` must walk every position a string can occupy.

It guards the audit log, trace store, lifecycle log, intent log and the pubsub
payload path (``src/host/mesh.py:930``), so anything it declines to walk is a
value that reaches durable storage unredacted. Three positions were skipped,
each demonstrated below against the pre-fix implementation:

  1. Dict KEYS — ``{k: deep_redact(v) ...}`` never touched ``k``.
  2. ``set`` / ``frozenset`` — fell through to ``return obj``, emitted verbatim.
  3. The URL gate — ``_looks_like_url`` required ``"://"``, but ``redact_url``
     itself also accepts a relative URL carrying a query string. The gate was
     stricter than the function it guarded, so those strings took the
     pattern-only path and kept their structural secrets.

``TestRedactionIsLinear`` covers the ReDoS that removing the gate's incidental
length cap exposed — see that class for why the bound is load-bearing.
"""

from __future__ import annotations

import time

import pytest

from src.shared.redaction import deep_redact

# Shaped like a real key so SECRET_PATTERNS matches it.
SECRET = "sk-ant-api03-REALSECRETVALUE1234567890abcdefghijklmnop"


def _leaks(value) -> bool:
    return SECRET in repr(value)


class TestSecretsInDictKeys:
    def test_secret_in_a_key_is_redacted(self):
        out = deep_redact({f"token={SECRET}": "v"})
        assert not _leaks(out), f"secret survived in a dict key: {out!r}"

    def test_secret_in_a_nested_key_is_redacted(self):
        out = deep_redact({"outer": [{f"{SECRET}": 1}]})
        assert not _leaks(out)

    def test_ordinary_keys_are_untouched(self):
        """Only secret-SHAPED keys change — the log stays readable."""
        assert deep_redact({"api_key_name": "x", "user_id": 7}) == {
            "api_key_name": "x",
            "user_id": 7,
        }

    def test_non_string_keys_survive(self):
        assert deep_redact({1: "a", (2, 3): "b"}) == {1: "a", (2, 3): "b"}


class TestUnhandledContainers:
    @pytest.mark.parametrize("factory", [set, frozenset, list, tuple])
    def test_container_contents_are_redacted(self, factory):
        out = deep_redact({"c": factory([SECRET])})
        assert not _leaks(out), f"secret survived inside a {factory.__name__}"

    @pytest.mark.parametrize(
        ("factory", "expected"),
        [(set, set), (frozenset, frozenset), (list, list), (tuple, tuple)],
    )
    def test_container_type_is_preserved(self, factory, expected):
        assert isinstance(deep_redact(factory(["plain"])), expected)


class TestUrlGate:
    def test_relative_url_with_query_gets_structural_redaction(self):
        out = deep_redact(f"/cb?code={SECRET}&state=ok")
        assert not _leaks(out)

    def test_scheme_relative_url_userinfo_is_stripped(self):
        """The gate's sharpest edge: NO ``://``, so URL structure was skipped.

        Pattern matching still caught the token, but ``user:hunter2`` in the
        userinfo is only reachable by parsing the URL — which the gate
        prevented.
        """
        out = deep_redact(f"//user:hunter2@example.com/cb?token={SECRET}")
        assert "hunter2" not in out, f"userinfo survived: {out!r}"
        assert not _leaks(out)

    def test_absolute_url_still_redacted(self):
        out = deep_redact(f"https://example.com/cb?token={SECRET}")
        assert not _leaks(out)

    def test_plain_text_passes_through(self):
        assert deep_redact("just a routine status line") == (
            "just a routine status line"
        )


class TestScalarsPassThrough:
    @pytest.mark.parametrize("value", [1, 1.5, True, False, None])
    def test_scalar_unchanged(self, value):
        assert deep_redact(value) is value


class TestRedactionIsLinear:
    """The sanitizer must not be a DoS vector.

    ``_CONN_USERINFO_RE`` used an UNBOUNDED scheme class
    (``[A-Za-z][A-Za-z0-9+.-]*://``). The engine matched it greedily from every
    position in the subject, consuming the rest of the string before failing to
    find ``://`` and backtracking over every length — quadratic in the subject
    length, on a sanitizer that runs over agent-supplied pubsub payloads and
    blocker notes on the mesh event loop. A 200 KB URL-shaped string took ~93
    SECONDS; an agent could stall the mesh by publishing one.

    ``deep_redact`` previously skipped URL parsing above 4096 chars, which
    incidentally hid part of this. That cap is gone (it was also what made the
    gate diverge from ``redact_url``), so the bound has to be real.
    """

    @staticmethod
    def _elapsed_ms(n: int) -> float:
        subject = "https://example.com/?q=" + "x" * n
        start = time.perf_counter()
        deep_redact(subject)
        return (time.perf_counter() - start) * 1000

    def test_large_url_shaped_string_is_fast(self):
        elapsed = self._elapsed_ms(200_000)
        assert elapsed < 2000, (
            f"deep_redact took {elapsed:.0f} ms on a 200 KB URL-shaped string. "
            "Before the scheme class was bounded this was ~93,000 ms — check "
            "for an unbounded quantifier in the redaction patterns."
        )

    def test_cost_grows_linearly_not_quadratically(self):
        """4x the input must not cost anything like 16x the time."""
        small = max(self._elapsed_ms(25_000), 0.5)
        large = self._elapsed_ms(100_000)
        assert large / small < 8, (
            f"4x input grew cost {large / small:.1f}x ({small:.1f} -> "
            f"{large:.1f} ms) — that is superlinear, i.e. backtracking."
        )
