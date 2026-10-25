"""Take the documentation screenshots from a running Web UI.

Run it against a Web UI started with a throwaway data folder and Hugging Face cache, never your own (see
docs/developer/docs.md#screenshots). Each shot writes ``docs/assets/screenshots/<name>.webp``.

    uv run --with playwright python scripts/docs_screenshots.py --url http://localhost:8790 [--only gallery-page,...]
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path

from PIL import Image
from playwright.async_api import Browser, Page, async_playwright

_OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "screenshots"
_DESKTOP = {"width": 1440, "height": 900}
_PHONE = {"width": 390, "height": 844}
_SCALE = 1.5
# The newest output: the image the Workspace and viewer shots show. Generated with the enhancer for the docs.
_HERO = "astronaut_2026-10-25_13-18-32.png"
_DEMO_PROMPT_FILE = "/var/tmp/ziv-docs/data/prompts/demo.yaml"


def _save(png: bytes, name: str) -> None:
    """Write a screenshot as WebP under its shot name."""
    _OUT.mkdir(parents=True, exist_ok=True)
    Image.open(io.BytesIO(png)).save(_OUT / f"{name}.webp", "WEBP", quality=85, method=6)
    print(f"wrote {name}.webp")


async def _page(browser: Browser, url: str, *, phone: bool = False) -> Page:
    """Open *url* in a fresh browser context, so no storage carries over between shots."""
    if phone:
        context = await browser.new_context(viewport=_PHONE, device_scale_factor=3, is_mobile=True, has_touch=True)
    else:
        context = await browser.new_context(viewport=_DESKTOP, device_scale_factor=_SCALE)
    page = await context.new_page()
    await page.goto(url, wait_until="networkidle")
    await page.wait_for_timeout(1200)
    return page


def _delete_new_outputs(url: str, before: set[str]) -> None:
    """Delete the outputs made since *before* through the app, so the hero stays the newest image."""
    with urllib.request.urlopen(f"{url}/api/gallery?page=1") as response:
        assets = json.load(response).get("assets", [])
    for asset in assets:
        if asset["id"] not in before:
            request = urllib.request.Request(f"{url}/api/gallery/{asset['id']}", method="DELETE", headers={"Origin": url})
            urllib.request.urlopen(request).read()


def _output_ids(url: str) -> set[str]:
    """Return the ids of the outputs on the Gallery's first page."""
    with urllib.request.urlopen(f"{url}/api/gallery?page=1") as response:
        return {asset["id"] for asset in json.load(response).get("assets", [])}


def _post_json(url: str, body: dict) -> None:
    """POST *body* to *url* as JSON, ignoring a conflict for something that already exists."""
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json", "Origin": url.split("/api/")[0]})
    try:
        urllib.request.urlopen(request).read()
    except urllib.error.HTTPError as err:
        if err.code != 409:
            raise


async def _dismiss_toasts(page: Page) -> None:
    """Close any toast, so it doesn't cover the shot."""
    for button in await page.locator('[aria-label="Dismiss"], [aria-label="Dismiss notification"], [aria-label="Close notification"]').all():
        await button.click(force=True)
    await page.wait_for_timeout(600)


async def shot_job_running(browser: Browser, url: str) -> None:
    """Start a job and capture it at 75%: the live preview beside the job card, and the card alone."""
    before = _output_ids(url)
    page = await _page(browser, url)
    await page.get_by_placeholder("Describe the scene...").fill("A lighthouse on a cliff at golden hour, waves below, seabirds, warm light")
    await page.get_by_role("button", name="Generate").last.click()
    await page.get_by_text("Live preview · 75%").wait_for(timeout=180_000)
    _save(await page.screenshot(), "job-running")
    _save(await page.screenshot(clip={"x": 1015, "y": 160, "width": 417, "height": 368}), "job-card")
    await page.get_by_text("Generating image").wait_for(state="detached", timeout=180_000)
    await page.wait_for_timeout(2000)
    _delete_new_outputs(url, before)


async def shot_job_log(browser: Browser, url: str) -> None:
    """Cancel a job so the log has a cancelled entry beside finished ones, then open the Job log."""
    before = _output_ids(url)
    page = await _page(browser, url)
    await page.get_by_placeholder("Describe the scene...").fill("A paper boat on a puddle after the rain, city reflections")
    await page.get_by_role("button", name="Generate").last.click()
    await page.get_by_text("Generating image").wait_for(timeout=180_000)
    await page.keyboard.press("Escape")
    await page.keyboard.press("Escape")
    await page.get_by_text("Generating image").wait_for(state="detached", timeout=180_000)
    await page.wait_for_timeout(1500)
    _delete_new_outputs(url, before)
    page = await _page(browser, url)
    await page.keyboard.press("g")
    await page.keyboard.press("j")
    await page.wait_for_timeout(1200)
    _save(await page.screenshot(clip={"x": 928, "y": 0, "width": 512, "height": 420}), "job-log")


async def shot_workspace(browser: Browser, url: str) -> None:
    page = await _page(browser, url)
    await page.locator(f'[aria-label^="Reuse settings from {_HERO}"]').first.click(force=True)
    await page.wait_for_timeout(1200)
    await page.get_by_role("tab", name="Enhanced").first.click()
    await _dismiss_toasts(page)
    _save(await page.screenshot(), "workspace")


async def shot_gallery_page(browser: Browser, url: str) -> None:
    page = await _page(browser, f"{url}/#/gallery")
    await page.wait_for_timeout(1500)
    _save(await page.screenshot(), "gallery-page")


async def shot_viewer_details(browser: Browser, url: str) -> None:
    page = await _page(browser, f"{url}/#/gallery")
    await page.wait_for_timeout(1500)
    await page.locator(f'[aria-label="View {_HERO}"]').first.click(force=True)
    await page.wait_for_timeout(1800)
    await page.keyboard.press("i")
    await page.wait_for_timeout(900)
    _save(await page.screenshot(), "viewer-details")


async def shot_models_page(browser: Browser, url: str) -> None:
    page = await _page(browser, f"{url}/#/models")
    await page.wait_for_timeout(2500)
    _save(await page.screenshot(), "models-page")


async def shot_config_page(browser: Browser, url: str) -> None:
    page = await _page(browser, f"{url}/#/config")
    await page.wait_for_timeout(1500)
    _save(await page.screenshot(), "config-page")


async def shot_prompts_page(browser: Browser, url: str) -> None:
    page = await _page(browser, f"{url}/#/prompts?path={_DEMO_PROMPT_FILE}")
    await page.wait_for_timeout(2000)
    _save(await page.screenshot(), "prompts-page")


async def shot_preset_menu(browser: Browser, url: str) -> None:
    for name, steps in (("Quick look", 4), ("Final render", 8)):
        _post_json(f"{url}/api/presets", {"name": name, "mode": "image", "model": "klein4b", "settings": {"steps": steps}})
    page = await _page(browser, url)
    await page.get_by_text("Preset", exact=True).first.click()
    await page.wait_for_timeout(700)
    _save(await page.screenshot(clip={"x": 0, "y": 0, "width": 720, "height": 420}), "preset-menu")


async def shot_phone_workspace(browser: Browser, url: str) -> None:
    page = await _page(browser, url, phone=True)
    await page.wait_for_timeout(1500)
    _save(await page.screenshot(), "phone-workspace")


async def shot_first_run_dialog(browser: Browser, url: str) -> None:
    """Capture the welcome dialog. Run this shot against a second Web UI with an empty data folder and cache."""
    page = await _page(browser, url)
    await page.wait_for_timeout(1500)
    _save(await page.screenshot(), "first-run-dialog")


SHOTS: dict[str, Callable[[Browser, str], Awaitable[None]]] = {
    "workspace": shot_workspace,
    "gallery-page": shot_gallery_page,
    "viewer-details": shot_viewer_details,
    "models-page": shot_models_page,
    "config-page": shot_config_page,
    "prompts-page": shot_prompts_page,
    "preset-menu": shot_preset_menu,
    "phone-workspace": shot_phone_workspace,
    "job-running": shot_job_running,
    "job-log": shot_job_log,
}

# Shots that need a Web UI whose data folder and cache are empty, run with --empty-url.
EMPTY_SHOTS: dict[str, Callable[[Browser, str], Awaitable[None]]] = {
    "first-run-dialog": shot_first_run_dialog,
}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="Base URL of the running Web UI")
    parser.add_argument("--empty-url", default="", help="Base URL of a second Web UI with an empty data folder and cache, for the first-run shot")
    parser.add_argument("--only", default="", help="Comma-separated shot names; all when empty")
    parser.add_argument("--chrome", default="/usr/bin/google-chrome", help="Chrome or Chromium executable")
    args = parser.parse_args()
    wanted = [name for name in args.only.split(",") if name]
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=args.chrome, headless=True)
        for name, shot in SHOTS.items():
            if not wanted or name in wanted:
                await shot(browser, args.url.rstrip("/"))
        for name, shot in EMPTY_SHOTS.items():
            if args.empty_url and (not wanted or name in wanted):
                await shot(browser, args.empty_url.rstrip("/"))
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
