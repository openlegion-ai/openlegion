"""``deep_redact`` must walk every position a string can occupy.

It guards the trace store, lifecycle log, intent log and the durable DM thread
record (``src/host/mesh.py:930`` — delivery keeps the original payload), so
anything it declines to walk reaches durable storage unredacted. It does NOT
guard ``Blackboard.log_audit``, which writes values verbatim; that is a
separate, still-open gap. Three positions were skipped,
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
        userinfo is only reachable by parsing the URL.
        """
        out = deep_redact(f"//user:hunter2@example.com/cb?token={SECRET}")
        assert "hunter2" not in out, f"userinfo survived: {out!r}"
        assert not _leaks(out)

    def test_network_path_userinfo_without_a_query(self):
        """Independent of query handling.

        ``redact_url``'s own gate accepted a relative ref only when it began
        with ``/`` AND contained ``?``, so a network-path reference carrying
        userinfo but no query fell through to pattern-only redaction.
        """
        out = deep_redact("//user:hunter2@example.com/cb")
        assert "hunter2" not in out, f"userinfo survived: {out!r}"

    def test_relative_url_fragment_is_dropped(self):
        """Fragments carry OAuth implicit tokens (``#access_token=…``).

        They were already dropped for absolute URLs; a relative ref with a
        fragment and no query was not parsed at all.
        """
        out = deep_redact("/cb#access_token=opaque-token-value")
        assert "opaque-token-value" not in out, f"fragment survived: {out!r}"

    @pytest.mark.parametrize(
        "path", ["/etc/passwd", "/var/log/app.log", "a/b/c", "C:\\path\\file"],
    )
    def test_plain_paths_are_not_reshaped(self, path: str):
        """A path with neither query nor fragment has nothing structural to
        strip, so it must pass through byte-identical."""
        assert deep_redact(path) == path

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


class TestLongCredentialsAreStillRedacted:
    """The ReDoS fix must not cap the userinfo runs.

    Bounding them to ``{1,256}`` fixed the ReDoS equally well but silently
    STOPPED redacting long credentials: a 900-char password matched with the
    run unbounded and was MISSED under the cap. That shape is real — AWS
    documents IAM database-auth tokens at roughly 1 KiB, and they are used
    directly as the password in a DSN.

    Only the SCHEME is bounded; see the comment on ``_CONN_USERINFO_RE``.
    """

    @pytest.mark.parametrize("length", [10, 256, 257, 900, 4000])
    def test_dsn_password_of_any_length_is_redacted(self, length: int):
        password = "p" * length
        out = deep_redact(f"postgres://dbuser:{password}@db.internal:5432/main")
        assert password not in out, (
            f"{length}-char DSN password survived — check for a capped "
            f"quantifier on the userinfo run: {out[:120]!r}"
        )

    @pytest.mark.parametrize("length", [256, 257, 900])
    def test_dsn_username_of_any_length_is_redacted(self, length: int):
        user = "u" * length
        out = deep_redact(f"postgres://{user}:pw@db/main")
        assert user not in out, f"{length}-char DSN username survived"


class TestBlockerNoteIsLinear:
    """``normalize_blocker_note`` runs on agent-controlled text, on the loop.

    ``_EMPTY_EXCEPTION_RE`` was ``^(exception:)?\\s*(error:?)?\\s*$`` — two
    ``\\s*`` runs either side of an optional group that can match empty, so on
    a long whitespace run the engine tried every split between them. The
    blocker note arrives on a task-status POST and is normalized synchronously
    BEFORE the 500-char truncation, so an agent could stall the mesh:
    "exception:" + 80k spaces took 34 SECONDS.
    """

    def test_long_whitespace_note_is_fast(self):
        from src.shared.redaction import normalize_blocker_note

        subject = "exception:" + " " * 80_000 + "x"
        start = time.perf_counter()
        normalize_blocker_note(subject)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 1000, (
            f"normalize_blocker_note took {elapsed:.0f} ms on 80k spaces "
            "(was ~34,000 ms) — check for ambiguous adjacent \\s* runs."
        )

    @pytest.mark.parametrize(
        ("note", "expected"),
        [
            ("exception:", "internal_error"),
            ("Exception: Error:", "internal_error"),
            ("Error:", "internal_error"),
            ("error", "internal_error"),
            ("", None),
            ("   ", None),
            ("exception: boom", "exception: boom"),
        ],
    )
    def test_classification_unchanged(self, note, expected):
        """The rewrite must classify exactly as before."""
        from src.shared.redaction import normalize_blocker_note

        assert normalize_blocker_note(note) == expected
