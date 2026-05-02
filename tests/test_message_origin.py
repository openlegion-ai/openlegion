"""Tests for the typed ``MessageOrigin`` model and ``trace.py`` helpers.

Task 2a (operator orchestration roadmap) introduces ``MessageOrigin`` as
a typed Pydantic model carrying ``kind``, ``channel``, and ``user``.

The test surface here covers three concerns:

* The model itself — validation, defaults, immutability, dict-compat.
* The ``X-Origin`` wire format — JSON shape, additive ``kind`` segment,
  legacy back-compat (no ``kind`` → ``kind="agent"``).
* The ``parse_origin_header`` / ``origin_header`` helpers in
  ``src.shared.trace`` — accepts both typed and dict input, returns
  typed output.

The "legacy header → kind=agent" case is the security-critical default:
unauthenticated paths must NOT be able to claim ``kind="human"`` by
sending the pre-Task-2a header shape.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.shared.trace import origin_header, parse_origin_header
from src.shared.types import MessageOrigin

# ── MessageOrigin model ─────────────────────────────────────────


class TestMessageOriginModel:
    def test_construction_with_all_fields(self):
        m = MessageOrigin(kind="human", channel="cli", user="jeff")
        assert m.kind == "human"
        assert m.channel == "cli"
        assert m.user == "jeff"

    def test_default_channel_and_user_are_empty_strings(self):
        m = MessageOrigin(kind="cron")
        assert m.channel == ""
        assert m.user == ""

    @pytest.mark.parametrize(
        "kind",
        ["human", "operator", "agent", "system", "heartbeat", "cron"],
    )
    def test_all_documented_kinds_accepted(self, kind):
        m = MessageOrigin(kind=kind)
        assert m.kind == kind

    @pytest.mark.parametrize("kind", ["", "user", "HUMAN", "haxxor", "admin"])
    def test_unknown_kind_rejected(self, kind):
        with pytest.raises(ValidationError):
            MessageOrigin(kind=kind)

    def test_frozen_blocks_mutation(self):
        m = MessageOrigin(kind="human", channel="cli", user="jeff")
        with pytest.raises(ValidationError):
            m.kind = "agent"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            m.channel = "telegram"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            m.user = "+1"  # type: ignore[misc]

    def test_dict_style_getitem(self):
        m = MessageOrigin(kind="human", channel="cli", user="jeff")
        assert m["kind"] == "human"
        assert m["channel"] == "cli"
        assert m["user"] == "jeff"
        with pytest.raises(KeyError):
            _ = m["nope"]

    def test_dict_style_get(self):
        m = MessageOrigin(kind="cron")
        # Empty channel/user return ``""`` (matches dict semantics where
        # the key is present but the value is empty).
        assert m.get("channel") == ""
        assert m.get("user") == ""
        assert m.get("kind") == "cron"
        # Unknown key falls back to default.
        assert m.get("nope") is None
        assert m.get("nope", "fallback") == "fallback"

    def test_dict_conversion_roundtrip(self):
        # ``lanes.py`` does ``dict(task.origin)`` — must produce a plain
        # dict containing all three fields.
        m = MessageOrigin(kind="operator", channel="dashboard", user="op-1")
        d = dict(m)
        assert d == {"kind": "operator", "channel": "dashboard", "user": "op-1"}


# ── X-Origin wire format ────────────────────────────────────────


class TestHeaderRoundTrip:
    def test_round_trip_preserves_all_fields(self):
        original = MessageOrigin(kind="human", channel="telegram", user="+99")
        wire = original.to_header_value()
        restored = MessageOrigin.from_header_value(wire, trust_kind=True)
        assert restored == original

    def test_to_header_value_emits_json(self):
        m = MessageOrigin(kind="human", channel="cli", user="jeff")
        # JSON shape lets old mesh nodes parse it through the legacy
        # ``channel``/``user`` whitelist; ``kind`` falls through.
        parsed = json.loads(m.to_header_value())
        assert parsed == {"kind": "human", "channel": "cli", "user": "jeff"}

    def test_from_header_value_none_or_empty_returns_none(self):
        assert MessageOrigin.from_header_value(None) is None
        assert MessageOrigin.from_header_value("") is None

    def test_from_header_value_malformed_returns_none_not_partial(self):
        # Malformed → ``None``, never a half-built model.
        assert MessageOrigin.from_header_value("not-json") is None
        assert MessageOrigin.from_header_value("{") is None
        assert MessageOrigin.from_header_value('"just-a-string"') is None
        assert MessageOrigin.from_header_value("[1, 2, 3]") is None
        assert MessageOrigin.from_header_value("null") is None

    def test_from_header_value_oversized_returns_none(self):
        big = '{"kind":"human","channel":"x","user":"' + ("y" * 600) + '"}'
        assert MessageOrigin.from_header_value(big) is None

    def test_from_header_value_extra_fields_dropped(self):
        # Whitelist-only — extras must not survive onto the model.
        result = MessageOrigin.from_header_value(
            '{"kind":"human","channel":"cli","user":"jeff","extra":"dropped"}'
        )
        assert result is not None
        assert not hasattr(result, "extra")


# ── Security-critical: legacy header default ────────────────────


class TestLegacyHeaderSecurityDefault:
    """A legacy ``X-Origin`` header (no ``kind``) MUST default to
    ``kind="agent"`` (least-trusted). This is the load-bearing security
    property — without it, an unauthenticated path could forge a
    ``human``-origin claim simply by sending the pre-Task-2a header.
    """

    def test_legacy_header_defaults_to_kind_agent(self):
        result = MessageOrigin.from_header_value(
            '{"channel":"cli","user":"jeff"}'
        )
        assert result is not None
        assert result.kind == "agent"
        assert result.channel == "cli"
        assert result.user == "jeff"

    def test_untrusted_typed_header_downgrades_privileged_kind(self):
        result = MessageOrigin.from_header_value(
            '{"kind":"human","channel":"cli","user":"jeff"}'
        )
        assert result is not None
        assert result.kind == "agent"

    def test_trusted_typed_header_preserves_stamped_kind(self):
        result = MessageOrigin.from_header_value(
            '{"kind":"human","channel":"cli","user":"jeff"}',
            trust_kind=True,
        )
        assert result is not None
        assert result.kind == "human"

    def test_legacy_header_with_empty_channel_or_user_rejected(self):
        # Pre-Task-2a parser rejected empty channel/user; preserve that
        # for legacy headers so half-shaped origins never reach
        # auto-notify.
        assert MessageOrigin.from_header_value(
            '{"channel":"","user":"jeff"}'
        ) is None
        assert MessageOrigin.from_header_value(
            '{"channel":"cli","user":""}'
        ) is None

    def test_typed_header_allows_empty_channel_user(self):
        # Cron / system / heartbeat origins have no addressable user.
        # The typed parser must accept them.
        for kind in ("cron", "system", "heartbeat"):
            result = MessageOrigin.from_header_value(
                f'{{"kind":"{kind}","channel":"","user":""}}',
                trust_kind=True,
            )
            assert result is not None, kind
            assert result.kind == kind

    def test_rolling_deploy_safety_old_parser_can_drop_kind(self):
        """A new-format header parsed by an *old* parser (which only
        whitelists ``channel`` / ``user``) must yield a usable origin
        with the minimum-trust default ``kind="agent"``.

        Simulated by stripping ``kind`` from the new-format JSON and
        feeding the result back through ``from_header_value``.
        """
        new_format = MessageOrigin(
            kind="human", channel="cli", user="jeff",
        ).to_header_value()
        parsed = json.loads(new_format)
        parsed.pop("kind")
        legacy_blob = json.dumps(parsed, separators=(",", ":"))
        result = MessageOrigin.from_header_value(legacy_blob)
        assert result is not None
        assert result.kind == "agent"


# ── trace.parse_origin_header / origin_header helpers ───────────


class TestTraceHelpers:
    def test_parse_origin_header_returns_typed_model(self):
        result = parse_origin_header(
            '{"kind":"human","channel":"cli","user":"jeff"}'
        )
        assert isinstance(result, MessageOrigin)
        assert result.kind == "agent"

    def test_parse_origin_header_legacy_defaults_to_agent(self):
        result = parse_origin_header('{"channel":"cli","user":"jeff"}')
        assert isinstance(result, MessageOrigin)
        assert result.kind == "agent"

    def test_parse_origin_header_invalid_returns_none(self):
        assert parse_origin_header(None) is None
        assert parse_origin_header("") is None
        assert parse_origin_header("not-json") is None

    def test_origin_header_typed_input(self):
        m = MessageOrigin(kind="human", channel="cli", user="jeff")
        h = origin_header(m)
        assert "X-Origin" in h
        parsed = json.loads(h["X-Origin"])
        assert parsed == {"kind": "human", "channel": "cli", "user": "jeff"}

    def test_origin_header_dict_input_back_compat(self):
        # Stamps not yet migrated still pass dicts. Must serialize.
        h = origin_header({"kind": "human", "channel": "cli", "user": "jeff"})
        parsed = json.loads(h["X-Origin"])
        assert parsed == {"kind": "human", "channel": "cli", "user": "jeff"}

    def test_origin_header_typed_and_dict_produce_identical_payload(self):
        m = MessageOrigin(kind="operator", channel="dashboard", user="op-1")
        h_typed = origin_header(m)
        h_dict = origin_header(
            {"kind": "operator", "channel": "dashboard", "user": "op-1"}
        )
        assert h_typed == h_dict

    def test_origin_header_legacy_dict_no_kind_upgrades_to_agent(self):
        # Dict without ``kind`` must serialize with ``kind="agent"`` so
        # the next hop never sees a kind-less origin on the wire.
        h = origin_header({"channel": "cli", "user": "jeff"})
        parsed = json.loads(h["X-Origin"])
        assert parsed["kind"] == "agent"
        assert parsed["channel"] == "cli"
        assert parsed["user"] == "jeff"

    def test_origin_header_none_returns_empty_dict(self):
        assert origin_header(None) == {}
        assert origin_header({}) == {}

    def test_round_trip_through_helpers(self):
        m = MessageOrigin(kind="human", channel="telegram", user="+99")
        wire = origin_header(m)["X-Origin"]
        restored = parse_origin_header(wire)
        assert restored == MessageOrigin(kind="agent", channel="telegram", user="+99")


# ── Request-level integration with parse_origin_header ──────────


class _FakeRequestHeaders:
    """Minimal stand-in for ``Request.headers`` (case-insensitive get)."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._lower = {k.lower(): v for k, v in mapping.items()}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._lower.get(key.lower(), default)


class TestRequestHeaderParse:
    def test_legacy_request_header_yields_kind_agent(self):
        req_headers = _FakeRequestHeaders(
            {"x-origin": '{"channel":"cli","user":"jeff"}'},
        )
        origin = parse_origin_header(req_headers.get("x-origin"))
        assert isinstance(origin, MessageOrigin)
        assert origin.kind == "agent"
        assert origin.channel == "cli"
        assert origin.user == "jeff"

    def test_typed_request_header_yields_stamped_kind(self):
        req_headers = _FakeRequestHeaders(
            {"x-origin": '{"kind":"human","channel":"cli","user":"jeff"}'},
        )
        origin = parse_origin_header(req_headers.get("x-origin"))
        assert isinstance(origin, MessageOrigin)
        assert origin.kind == "agent"
