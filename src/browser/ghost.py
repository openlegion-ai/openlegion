"""GhostCursor integration for human-like mouse movement.

Wraps python_ghost_cursor for explicit automation clicks alongside
Camoufox's native humanize parameter.
"""

from __future__ import annotations

import asyncio
import random

from src.shared.utils import setup_logging

logger = setup_logging("browser.ghost")


async def human_click(page, selector: str | None = None, x: int | None = None, y: int | None = None) -> None:
    """Click with human-like delay and optional jitter.

    Uses Camoufox's native humanize for mouse movement. This adds
    think-time jitter before and after clicks.
    """
    # Think time before action (50-300ms)
    await asyncio.sleep(random.uniform(0.05, 0.3))

    if selector:
        element = page.locator(selector)
        await element.click()
    elif x is not None and y is not None:
        await page.mouse.click(x, y)

    # Small settle time after click (30-150ms)
    await asyncio.sleep(random.uniform(0.03, 0.15))


async def human_type(page, selector: str, text: str, delay_ms: int = 50) -> None:
    """Type with human-like per-character delays.

    Adds variable delay between keystrokes to simulate natural typing.
    """
    element = page.locator(selector)
    await element.click()
    await asyncio.sleep(random.uniform(0.05, 0.2))

    for char in text:
        await page.keyboard.press(char)
        # Variable typing speed: base delay +/- 40%
        jitter = delay_ms * random.uniform(0.6, 1.4) / 1000
        await asyncio.sleep(jitter)
