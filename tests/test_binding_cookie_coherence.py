"""Tests for the always-on bound-cookie identity-coherence check.

``BrowserManager._enforce_binding_cookie_coherence`` mirrors the opt-in
session-persistence sidecar's binding-drop onto the always-on Camoufox
persistent-profile channel. When the agent's ``(UA, proxy)`` signature
changed since last launch, anti-bot bound cookies (cf_clearance, datadome,
_abck, …) must be dropped BEFORE they replay under the new fingerprint —
otherwise an instant 403 + trust-score hit. Runs regardless of
``BROWSER_SESSION_PERSISTENCE_ENABLED``.

Covers:
  * mismatched persisted signature → matching bound cookies removed,
    non-bound (generic session) cookies retained; new signature persisted.
  * matching persisted signature → nothing dropped (cookies never read).
  * first run (no persisted signature) → nothing dropped, signature
    persisted so the NEXT identity change is detectable.
  * legacy Playwright without ``clear_cookies(name=...)`` → clear-all +
    re-add of the non-bound subset (same end state).

BC-4 retry safety — the baseline is advanced ONLY when the drop is verified,
otherwise the OLD signature is retained so the NEXT launch retries (never
advance over a failed drop, or poisoned cookies replay under the new identity):
  * mismatch + ``context.cookies()`` raises (enumeration failed) → baseline NOT
    advanced.
  * mismatch + drop returns 0 while bound cookies present → baseline NOT
    advanced.
  * mismatch + enumeration returns ``[]`` (no bound cookies) → baseline
    advanced.
  * mismatch + drop reports success but a bound cookie survives the post-drop
    readback → baseline NOT advanced.
"""

from __future__ import annotations

import tempfile
import types

import pytest

from src.browser import session_persistence as sp
from src.browser.service import BrowserManager


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Clean binding-signature global + a private state sidecar per test."""
    from src.browser.service import _binding_signatures

    monkeypatch.setenv(
        "FINGERPRINT_STATE_PATH",
        str(tmp_path / "fp_state.json"),
    )
    _binding_signatures.clear()
    yield
    _binding_signatures.clear()


def _cookie(name: str) -> dict:
    return {
        "name": name,
        "value": "v-" + name,
        "domain": ".example.com",
        "path": "/",
    }


# A realistic mix: three vendor-bound cookies + two generic session cookies.
_BOUND = ["cf_clearance", "datadome", "_abck"]
_GENERIC = ["session", "csrftoken"]


class _FakeContext:
    """Playwright BrowserContext duck-type supporting name-filtered clear.

    Faithfully mutates its cookie jar on clear/add so the post-drop readback
    (``_bound_cookies_gone``) reflects the real end state.
    """

    def __init__(self, cookies: list[dict]):
        self._cookies = list(cookies)
        self.cookies_calls = 0
        self.cleared_names: list[str] = []
        self.cleared_all = 0
        self.readded: list[dict] | None = None

    async def cookies(self) -> list[dict]:
        self.cookies_calls += 1
        return list(self._cookies)

    async def clear_cookies(self, name: str | None = None) -> None:
        if name is None:
            self.cleared_all += 1
            self._cookies = []
        else:
            self.cleared_names.append(name)
            self._cookies = [c for c in self._cookies if c.get("name") != name]

    async def add_cookies(self, cookies: list[dict]) -> None:
        self.readded = list(cookies)
        self._cookies = self._cookies + list(cookies)


class _LegacyContext:
    """Older Playwright: ``clear_cookies`` has NO ``name`` kwarg.

    Calling ``clear_cookies(name=...)`` therefore raises ``TypeError`` —
    the exact signal the coherence helper falls back on. Faithfully mutates
    its jar so the post-drop readback reflects the clear-all + re-add end state.
    """

    def __init__(self, cookies: list[dict]):
        self._cookies = list(cookies)
        self.cleared_all = 0
        self.readded: list[dict] | None = None

    async def cookies(self) -> list[dict]:
        return list(self._cookies)

    async def clear_cookies(self) -> None:  # no ``name`` param on purpose
        self.cleared_all += 1
        self._cookies = []

    async def add_cookies(self, cookies: list[dict]) -> None:
        self.readded = list(cookies)
        self._cookies = self._cookies + list(cookies)


class _RaisingCookiesContext:
    """``context.cookies()`` raises → enumeration fails (``cookies is None``)."""

    def __init__(self):
        self.cleared_names: list[str] = []
        self.cleared_all = 0

    async def cookies(self) -> list[dict]:
        raise RuntimeError("cookies() boom")

    async def clear_cookies(self, name: str | None = None) -> None:
        if name is None:
            self.cleared_all += 1
        else:
            self.cleared_names.append(name)

    async def add_cookies(self, cookies: list[dict]) -> None:  # pragma: no cover
        pass


class _BrokenClearContext:
    """Both clear paths raise → ``_clear_bound_cookies`` returns 0 (drop failed).

    ``clear_cookies(name=...)`` raises a non-``TypeError`` (so the helper takes
    the clear-all fallback), and the fallback ``clear_cookies()`` raises too, so
    the final count is 0 while bound cookies are still present.
    """

    def __init__(self, cookies: list[dict]):
        self._cookies = list(cookies)
        self.cookies_calls = 0

    async def cookies(self) -> list[dict]:
        self.cookies_calls += 1
        return list(self._cookies)

    async def clear_cookies(self, name: str | None = None) -> None:
        raise RuntimeError("clear-all boom" if name is None else "targeted boom")

    async def add_cookies(self, cookies: list[dict]) -> None:  # pragma: no cover
        pass


class _NoOpClearContext:
    """``clear_cookies(name=...)`` SUCCEEDS but removes nothing.

    The silent-no-op drop the post-drop readback guards against: the drop
    reports success (count > 0) yet the jar never shrinks, so a re-read still
    finds the bound cookies.
    """

    def __init__(self, cookies: list[dict]):
        self._cookies = list(cookies)
        self.cookies_calls = 0
        self.cleared_names: list[str] = []
        self.cleared_all = 0

    async def cookies(self) -> list[dict]:
        self.cookies_calls += 1
        return list(self._cookies)  # never shrinks

    async def clear_cookies(self, name: str | None = None) -> None:
        if name is None:
            self.cleared_all += 1
        else:
            self.cleared_names.append(name)

    async def add_cookies(self, cookies: list[dict]) -> None:  # pragma: no cover
        pass


def _make_manager() -> BrowserManager:
    root = tempfile.mkdtemp(prefix="ol_bcc_")
    return BrowserManager(profiles_dir=root)


def _inst(agent_id: str, context) -> types.SimpleNamespace:
    return types.SimpleNamespace(agent_id=agent_id, context=context)


def _current_sig(mgr: BrowserManager, agent_id: str) -> str:
    return sp._hash_signature(mgr._build_session_binding_signature(agent_id))


class TestSignatureMismatch:
    @pytest.mark.asyncio
    async def test_mismatch_drops_bound_keeps_generic(self):
        from src.browser.service import _binding_signatures

        mgr = _make_manager()
        agent = "agent-mismatch"
        mgr.set_proxy_config(agent, {"server": "http://proxy-new:8080"})
        # Seed a DIFFERENT prior signature → forces the drop branch.
        _binding_signatures[agent] = "deadbeefdeadbeef"

        ctx = _FakeContext([_cookie(n) for n in _BOUND + _GENERIC])
        await mgr._enforce_binding_cookie_coherence(_inst(agent, ctx))

        # Every bound cookie was cleared by name; no generic cookie was.
        assert set(ctx.cleared_names) == set(_BOUND)
        assert all(g not in ctx.cleared_names for g in _GENERIC)
        assert ctx.cleared_all == 0  # targeted path, not clear-all
        # The current signature is now the persisted baseline.
        assert _binding_signatures[agent] == _current_sig(mgr, agent)


class TestSignatureMatch:
    @pytest.mark.asyncio
    async def test_match_drops_nothing_and_skips_cookie_read(self):
        from src.browser.service import _binding_signatures

        mgr = _make_manager()
        agent = "agent-match"
        mgr.set_proxy_config(agent, {"server": "http://proxy-stable:8080"})
        # Seed the SAME signature the agent will compute now.
        _binding_signatures[agent] = _current_sig(mgr, agent)

        ctx = _FakeContext([_cookie(n) for n in _BOUND + _GENERIC])
        await mgr._enforce_binding_cookie_coherence(_inst(agent, ctx))

        assert ctx.cleared_names == []
        assert ctx.cleared_all == 0
        # Unchanged identity ⇒ we don't even enumerate cookies.
        assert ctx.cookies_calls == 0
        # Signature is re-persisted (unchanged).
        assert _binding_signatures[agent] == _current_sig(mgr, agent)


class TestFirstRun:
    @pytest.mark.asyncio
    async def test_first_run_drops_nothing_but_persists_signature(self):
        from src.browser.service import _binding_signatures

        mgr = _make_manager()
        agent = "agent-first"
        mgr.set_proxy_config(agent, {"server": "http://proxy-first:8080"})
        assert agent not in _binding_signatures  # no prior baseline

        ctx = _FakeContext([_cookie(n) for n in _BOUND + _GENERIC])
        await mgr._enforce_binding_cookie_coherence(_inst(agent, ctx))

        assert ctx.cleared_names == []
        assert ctx.cleared_all == 0
        assert ctx.cookies_calls == 0  # nothing to invalidate on first run
        # But the baseline IS recorded so the next change is detectable.
        assert _binding_signatures[agent] == _current_sig(mgr, agent)


class TestLegacyPlaywrightFallback:
    @pytest.mark.asyncio
    async def test_fallback_clears_all_then_readds_non_bound(self):
        from src.browser.service import _binding_signatures

        mgr = _make_manager()
        agent = "agent-legacy"
        mgr.set_proxy_config(agent, {"server": "http://proxy-legacy:8080"})
        _binding_signatures[agent] = "0000000000000000"  # force mismatch

        ctx = _LegacyContext([_cookie(n) for n in _BOUND + _GENERIC])
        await mgr._enforce_binding_cookie_coherence(_inst(agent, ctx))

        # Fell back to clear-all …
        assert ctx.cleared_all == 1
        # … then re-added ONLY the non-bound cookies.
        assert ctx.readded is not None
        readded_names = {c["name"] for c in ctx.readded}
        assert readded_names == set(_GENERIC)
        assert all(b not in readded_names for b in _BOUND)
        assert _binding_signatures[agent] == _current_sig(mgr, agent)


class TestRetrySafetyBC4:
    """BC-4: a mismatch that could not verify the drop must NOT advance the
    persisted baseline — the OLD signature stays so the next launch retries."""

    @pytest.mark.asyncio
    async def test_enumeration_failure_keeps_old_baseline(self):
        from src.browser.service import _binding_signatures

        mgr = _make_manager()
        agent = "agent-enum-fail"
        mgr.set_proxy_config(agent, {"server": "http://proxy-enum:8080"})
        stale = "deadbeefdeadbeef"
        _binding_signatures[agent] = stale  # force mismatch

        ctx = _RaisingCookiesContext()
        await mgr._enforce_binding_cookie_coherence(_inst(agent, ctx))

        # Enumeration raised → nothing dropped, and the OLD baseline is kept so
        # the next launch sees a mismatch again and retries.
        assert ctx.cleared_names == []
        assert ctx.cleared_all == 0
        assert _binding_signatures[agent] == stale
        assert _binding_signatures[agent] != _current_sig(mgr, agent)

    @pytest.mark.asyncio
    async def test_drop_failure_keeps_old_baseline(self):
        from src.browser.service import _binding_signatures

        mgr = _make_manager()
        agent = "agent-drop-fail"
        mgr.set_proxy_config(agent, {"server": "http://proxy-drop:8080"})
        stale = "cafecafecafecafe"
        _binding_signatures[agent] = stale  # force mismatch

        ctx = _BrokenClearContext([_cookie(n) for n in _BOUND + _GENERIC])
        await mgr._enforce_binding_cookie_coherence(_inst(agent, ctx))

        # Both clear paths raised → drop returned 0 while bound cookies remain,
        # so the baseline is NOT advanced (retry next launch).
        assert _binding_signatures[agent] == stale
        assert _binding_signatures[agent] != _current_sig(mgr, agent)

    @pytest.mark.asyncio
    async def test_successful_drop_advances_baseline(self):
        from src.browser.service import _binding_signatures

        mgr = _make_manager()
        agent = "agent-drop-ok"
        mgr.set_proxy_config(agent, {"server": "http://proxy-ok:8080"})
        _binding_signatures[agent] = "1111111111111111"  # force mismatch

        ctx = _FakeContext([_cookie(n) for n in _BOUND + _GENERIC])
        await mgr._enforce_binding_cookie_coherence(_inst(agent, ctx))

        # Verified drop (readback finds no bound family left) → baseline advances.
        assert set(ctx.cleared_names) == set(_BOUND)
        assert _binding_signatures[agent] == _current_sig(mgr, agent)

    @pytest.mark.asyncio
    async def test_no_bound_cookies_advances_baseline(self):
        from src.browser.service import _binding_signatures

        mgr = _make_manager()
        agent = "agent-empty"
        mgr.set_proxy_config(agent, {"server": "http://proxy-empty:8080"})
        _binding_signatures[agent] = "2222222222222222"  # force mismatch

        # Enumeration SUCCEEDS but returns an empty jar (distinct from a None
        # failure) — nothing bound to drop → safe to advance.
        ctx = _FakeContext([])
        await mgr._enforce_binding_cookie_coherence(_inst(agent, ctx))

        assert ctx.cookies_calls == 1  # enumerated once; no drop, no readback
        assert ctx.cleared_names == []
        assert ctx.cleared_all == 0
        assert _binding_signatures[agent] == _current_sig(mgr, agent)

    @pytest.mark.asyncio
    async def test_readback_detecting_surviving_bound_keeps_old_baseline(self):
        from src.browser.service import _binding_signatures

        mgr = _make_manager()
        agent = "agent-noop-drop"
        mgr.set_proxy_config(agent, {"server": "http://proxy-noop:8080"})
        stale = "3333333333333333"
        _binding_signatures[agent] = stale  # force mismatch

        # Drop reports success by name, but the jar never shrinks — the post-drop
        # readback still sees bound cookies → treat as NOT verified.
        ctx = _NoOpClearContext([_cookie(n) for n in _BOUND + _GENERIC])
        await mgr._enforce_binding_cookie_coherence(_inst(agent, ctx))

        assert set(ctx.cleared_names) == set(_BOUND)
        assert ctx.cookies_calls == 2  # enumeration + post-drop readback
        assert _binding_signatures[agent] == stale
        assert _binding_signatures[agent] != _current_sig(mgr, agent)
