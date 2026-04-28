"""Tests for CAPTCHA token injection result handling."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.browser.captcha import CaptchaSolver


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
async def test_inject_token_exception_returns_false():
    solver = CaptchaSolver("2captcha", "key")
    page = AsyncMock()
    page.evaluate = AsyncMock(side_effect=RuntimeError("js context closed"))

    assert await solver._inject_token(page, "recaptcha-v2-checkbox", "tok") is False
