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
