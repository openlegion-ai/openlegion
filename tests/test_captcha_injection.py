"""Tests for CAPTCHA token injection result handling."""

from __future__ import annotations

import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser.captcha import CaptchaSolver


def test_solver_constructor_does_not_require_current_event_loop():
    errors: list[BaseException] = []

    def build_solver() -> None:
        try:
            CaptchaSolver("2captcha", "key")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=build_solver)
    thread.start()
    thread.join()

    assert errors == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "captcha_type",
    [
        "recaptcha-v2-checkbox",
        "recaptcha-v3",
        "hcaptcha",
        "turnstile",
    ],
)
async def test_inject_token_requires_page_script_success(captcha_type):
    solver = CaptchaSolver("2captcha", "key")
    page = AsyncMock()
    page.evaluate = AsyncMock(return_value=False)

    assert await solver._inject_token(page, captcha_type, "tok") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "captcha_type",
    [
        "recaptcha-v2-checkbox",
        "recaptcha-v3",
        "hcaptcha",
        "turnstile",
    ],
)
async def test_inject_token_reports_page_script_success(captcha_type):
    solver = CaptchaSolver("2captcha", "key")
    page = AsyncMock()
    page.evaluate = AsyncMock(return_value=True)

    assert await solver._inject_token(page, captcha_type, "tok") is True


@pytest.mark.asyncio
async def test_inject_token_walks_child_frames_without_repeating_main_frame():
    solver = CaptchaSolver("2captcha", "key")
    page = MagicMock()
    page.evaluate = AsyncMock(return_value=False)
    main_frame = MagicMock()
    main_frame.evaluate = AsyncMock(return_value=True)
    child_frame = MagicMock()
    child_frame.evaluate = AsyncMock(return_value=True)
    page.main_frame = main_frame
    page.frames = [main_frame, child_frame]

    assert await solver._inject_token(page, "hcaptcha", "tok") is True
    page.evaluate.assert_awaited_once()
    main_frame.evaluate.assert_not_awaited()
    child_frame.evaluate.assert_awaited_once()


@pytest.mark.asyncio
async def test_inject_token_exception_returns_false():
    solver = CaptchaSolver("2captcha", "key")
    page = AsyncMock()
    page.evaluate = AsyncMock(side_effect=RuntimeError("js context closed"))

    assert await solver._inject_token(page, "recaptcha-v2-checkbox", "tok") is False


# ── B2 STEALTH — selector tightening per family ────────────────────────────


@pytest.mark.asyncio
async def test_turnstile_injection_requires_widget_context():
    """Turnstile injection must require BOTH the canonical
    ``[name="cf-turnstile-response"]`` field AND a real Turnstile widget
    context (closest ``.cf-turnstile`` ancestor or a
    ``challenges.cloudflare.com`` iframe). The pre-B2 substring fallback
    ``input[name*="turnstile"]`` was a false-positive vector — A/B test
    flags + marketing pixels with ``turnstile`` in unrelated pages would
    return ``updated=true`` and let us bill for an injection that landed
    nowhere.
    """
    from src.browser.captcha import CaptchaSolver
    solver = CaptchaSolver("2captcha", "key")
    # Inspect the JS source so the test doesn't need a real DOM —
    # the contract under test is "the injection JS does NOT match
    # ``name*="turnstile"`` substring inputs". We accept any
    # cf-turnstile-response-only impl.
    js_src = ""

    async def capture(js, *_args):
        nonlocal js_src
        if "turnstile" in js.lower() and "cf-turnstile" in js:
            js_src = js
        return False

    page = MagicMock()
    page.evaluate = AsyncMock(side_effect=capture)
    page.frames = []
    page.main_frame = MagicMock()

    await solver._inject_token(page, "turnstile", "tok")
    # The substring fallback is removed.
    assert 'input[name*="turnstile"]' not in js_src
    # The exact name selector remains.
    assert '[name="cf-turnstile-response"]' in js_src
    # Widget-context guard appears.
    assert ".cf-turnstile" in js_src


@pytest.mark.asyncio
async def test_hcaptcha_branch_does_not_touch_g_recaptcha_response():
    """B2: the hCaptcha branch previously also wrote into
    ``[name="g-recaptcha-response"]`` — that's a cross-family bug
    (g-recaptcha-response is a reCAPTCHA field). On pages embedding
    BOTH widgets it leaked the hCaptcha token into the reCAPTCHA
    response field; on pages with only an unrelated reCAPTCHA hidden
    input it silently flipped ``updated=true`` for unrelated DOM."""
    from src.browser.captcha import CaptchaSolver
    solver = CaptchaSolver("2captcha", "key")

    # Capture the JS source the hcaptcha branch evaluates.
    js_src = ""

    async def capture(js, *_args):
        nonlocal js_src
        if "h-captcha-response" in js:
            js_src = js
        return False

    page = MagicMock()
    page.evaluate = AsyncMock(side_effect=capture)
    page.frames = []
    page.main_frame = MagicMock()

    await solver._inject_token(page, "hcaptcha", "tok")
    # The hCaptcha branch must touch h-captcha-response …
    assert '[name="h-captcha-response"]' in js_src
    # … and must NOT touch g-recaptcha-response (reCAPTCHA's field).
    assert "g-recaptcha-response" not in js_src
