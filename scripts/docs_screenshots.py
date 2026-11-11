"""Take the documentation screenshots from a running Web UI.

Run it against a Web UI started with a throwaway data folder and Hugging Face cache, never your own (see
docs/developer/docs.md#screenshots). Each shot writes ``docs/assets/screenshots/<name>.webp``.

    uv run --with playwright python scripts/docs_screenshots.py --url http://localhost:8790 [--only gallery-page,...]

make-media and the job shots load models: they need a free GPU. Their results stay in the data folder as demo content.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import time
import urllib.error
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
_HERO = "prompt_2026-10-31_13-32-30.png"
_DATA_DIR = Path("/var/tmp/ziv-docs/data")  # the throwaway data folder of docs/developer/docs.md
_DEMO_PROMPT_FILE = _DATA_DIR / "prompts" / "demo.yaml"
_STAMP = re.compile(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.")


async def _save(page: Page, name: str, clip: dict[str, float] | None = None) -> None:
    """Write a screenshot of *page* as WebP under its shot name, cropped to *clip* (CSS pixels) when given."""
    _OUT.mkdir(parents=True, exist_ok=True)
    png = await page.screenshot(clip=clip)
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


def _clear_job_log(url: str) -> None:
    """Empty the job log through the app, so the Job log shot shows only the jobs these shots run."""
    request = urllib.request.Request(f"{url}/api/jobs/log", method="DELETE", headers={"Origin": url})
    urllib.request.urlopen(request).read()


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
    await page.mouse.move(1000, 110)  # off every tile, so none shows its hover buttons
    await page.wait_for_timeout(600)


async def shot_job_running(browser: Browser, url: str) -> None:
    """Start a job and capture it at 75%: the live preview beside the job card, and the card alone."""
    before = _output_ids(url)
    _clear_job_log(url)
    page = await _page(browser, url)
    await page.get_by_placeholder("Describe the scene...").fill("A lighthouse on a cliff at golden hour, waves below, seabirds, warm light")
    await page.get_by_role("button", name="Generate").last.click()
    await page.get_by_text("Live preview · 75%").wait_for(timeout=180_000)
    await _save(page, "job-running")
    await _save(page, "job-card", {"x": 1015, "y": 160, "width": 417, "height": 368})
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
    await _save(page, "job-log", {"x": 928, "y": 0, "width": 512, "height": 420})


async def shot_workspace(browser: Browser, url: str) -> None:
    page = await _page(browser, url)
    await page.locator(f'[aria-label^="Reuse settings from {_HERO}"]').first.click(force=True)
    await page.wait_for_timeout(1200)
    await page.get_by_role("tab", name="Enhanced").first.click()
    await _dismiss_toasts(page)
    await _save(page, "workspace")


async def shot_enhance_popover(browser: Browser, url: str) -> None:
    """Capture the Enhance popover with a Shot, an Angle and a Palette chosen. No enhancement runs."""
    page = await _page(browser, url)
    await page.locator(f'[aria-label^="Reuse settings from {_HERO}"]').first.click(force=True)
    await page.wait_for_timeout(1200)
    await _dismiss_toasts(page)
    await page.locator("#ws-prompt").fill("A lone astronaut standing among glowing flowers on an alien planet")
    await page.locator("#ws-enhance-toggle").click()
    await page.wait_for_timeout(700)
    for key, slug in (("shot", "wide"), ("angle", "low"), ("palette", "earth")):
        await page.locator(f"#ws-enhance-inline-{key}").click()
        await page.locator(f'[data-slug="{slug}"]').first.click()
        await page.wait_for_timeout(300)
    await page.evaluate("document.activeElement instanceof HTMLElement && document.activeElement.blur()")
    await page.wait_for_timeout(700)
    panel = await page.locator("#ws-enhance-panel").bounding_box()
    assert panel is not None
    await _save(page, "enhance-popover", panel)


async def shot_gallery_page(browser: Browser, url: str) -> None:
    page = await _page(browser, f"{url}/#/gallery")
    await page.wait_for_timeout(1500)
    await _save(page, "gallery-page")


async def shot_viewer_details(browser: Browser, url: str) -> None:
    page = await _page(browser, f"{url}/#/gallery")
    await page.wait_for_timeout(1500)
    await page.locator(f'[aria-label="View {_HERO}"]').first.click(force=True)
    await page.wait_for_timeout(1800)
    await page.keyboard.press("i")
    await page.wait_for_timeout(900)
    await _save(page, "viewer-details")


async def shot_models_page(browser: Browser, url: str) -> None:
    page = await _page(browser, f"{url}/#/models")
    await page.wait_for_timeout(2500)
    await _save(page, "models-page", {"x": 0, "y": 0, "width": 1440, "height": 710})


async def shot_config_page(browser: Browser, url: str) -> None:
    page = await _page(browser, f"{url}/#/config")
    await page.wait_for_timeout(1500)
    await _save(page, "config-page")


async def shot_prompts_page(browser: Browser, url: str) -> None:
    page = await _page(browser, f"{url}/#/prompts?path={_DEMO_PROMPT_FILE}")
    await page.wait_for_timeout(2000)
    await _save(page, "prompts-page")


async def shot_preset_menu(browser: Browser, url: str) -> None:
    for name, steps in (("Quick look", 4), ("Final render", 8)):
        _post_json(f"{url}/api/presets", {"name": name, "mode": "image", "model": "klein4b", "settings": {"steps": steps}})
    page = await _page(browser, url)
    await page.get_by_text("Preset", exact=True).first.click()
    await page.wait_for_timeout(700)
    await _save(page, "preset-menu", {"x": 0, "y": 0, "width": 720, "height": 420})


async def shot_phone_workspace(browser: Browser, url: str) -> None:
    """Capture the Workspace on a phone with the hero's prompt and settings loaded."""
    page = await _page(browser, url, phone=True)
    await page.get_by_label(re.compile("Reuse settings")).first.click()
    await page.wait_for_timeout(1500)
    await _dismiss_toasts(page)
    # The strip scrolls sideways; bring the Images chip into view.
    await page.locator('[data-testid="settings-strip"] .setchip[aria-label^="Images"]').scroll_into_view_if_needed()
    await page.wait_for_timeout(500)
    await _save(page, "phone-workspace")


async def shot_first_run_dialog(browser: Browser, url: str) -> None:
    """Capture the welcome dialog. Run this shot against a second Web UI with an empty data folder and cache."""
    page = await _page(browser, url)
    await page.wait_for_timeout(1500)
    await _save(page, "first-run-dialog")


# --- Media shots ------------------------------------------------------------------------------------------------------
# The demo outputs are ordered by their timestamps, oldest first, with the hero newest; the takes and the clip made by the
# job shots slot in by their own timestamps, so the Gallery strip shows every kind of media. A shot that wants one of them
# in the preview moves it to the front first.


def _stamp(name: str) -> str:
    """Return the timestamp a file name carries, or an empty string."""
    match = _STAMP.search(name)
    return match.group(1) if match else ""


def _order_outputs(newest: str | None = None) -> None:
    """Set the modification times so the Gallery lists the hero first, or *newest* when one is named."""

    def key(path: Path) -> tuple[int, str]:
        return (2 if path.name == newest else 1 if path.name == _HERO else 0), _stamp(path.name)

    files = [path for path in (_DATA_DIR / "outputs").iterdir() if path.is_file()]
    start = time.time() - 600
    for index, path in enumerate(sorted(files, key=key)):
        os.utime(path, (start + index * 5, start + index * 5))


def _assets(url: str) -> list[dict]:
    """Return every Gallery asset, newest first."""
    assets: list[dict] = []
    page = 1
    while True:
        with urllib.request.urlopen(f"{url}/api/gallery?page={page}") as response:
            body = json.load(response)
        assets += body.get("assets", [])
        if page >= body.get("total_pages", 1):
            return assets
        page += 1


def _newest(url: str, workflow: str) -> str | None:
    """Return the file name of the newest asset made by *workflow* (a Gallery workflow: music, speech, sound, txt2vid)."""
    for asset in sorted(_assets(url), key=lambda asset: _stamp(asset["filename"]), reverse=True):
        if asset["workflow"] == workflow:
            return asset["filename"]
    return None


def _job_state(url: str) -> dict:
    """Return the active job and the queued ones."""
    with urllib.request.urlopen(f"{url}/api/jobs") as response:
        return json.load(response)


def _last_job_status(url: str) -> str:
    """Return the outcome of the newest entry in the job log."""
    with urllib.request.urlopen(f"{url}/api/jobs/log?page=1") as response:
        entries = json.load(response).get("entries") or []
    return str(entries[0].get("status")) if entries else "none"


async def _wait_job(url: str, *, started_within: float = 90, timeout: float = 1800) -> None:
    """Wait for a job to start and then for the queue to empty; raise when the newest job did not complete."""
    deadline = time.time() + started_within
    while not _job_state(url).get("active_job"):
        if time.time() > deadline:
            raise RuntimeError("the job never started")
        await asyncio.sleep(1)
    deadline = time.time() + timeout
    while _job_state(url).get("active_job") or _job_state(url).get("queued_jobs"):
        if time.time() > deadline:
            raise RuntimeError("the job did not finish in time")
        await asyncio.sleep(3)
    status = _last_job_status(url)
    if status != "completed":
        raise RuntimeError(f"the job ended as {status}")


async def _blur(page: Page) -> None:
    """Take the focus off the field being typed in, so no caret or focus ring shows."""
    await page.evaluate("document.activeElement instanceof HTMLElement && document.activeElement.blur()")
    await page.mouse.move(1000, 110)
    await page.wait_for_timeout(700)


async def _open_audio(page: Page, kind: str, model: str | None = None) -> None:
    """Open the Audio tab on *kind* (Music, Speech or Sound), choosing *model* by a part of its label when given."""
    await page.get_by_role("button", name="Audio", exact=True).click()
    await page.wait_for_timeout(600)
    await page.get_by_role("button", name=kind, exact=True).click()
    await page.wait_for_timeout(700)
    if model is not None:
        await page.locator("#ws-model").click()
        await page.get_by_role("option", name=re.compile(model)).click()
        await page.wait_for_timeout(700)


_SPEECH_TEXT = "Welcome back. It's good to see you again."
_SPEECH_LONG = (
    "Good evening, and welcome to the harbour. Tonight the tide comes in at nine, the lamps are lit along the pier, "
    "and the last ferry leaves just after ten. Take your time, find a seat by the water, and listen for the bells."
)
_STYLE = "Warm acoustic folk, fingerpicked guitar and soft piano, gentle female vocals, unhurried"
_VERSE = "Snow on the window, the kettle is low\nI keep your old sweater, I never let go"
_CHORUS = "Hold on, hold on, till the morning comes\nWe'll sing it softly, we'll sing it as one"


async def shot_audio_music(browser: Browser, url: str) -> None:
    """Capture the Audio tab on Music with a Style, two song blocks and a tempo, and a music take in the preview."""
    _order_outputs(_newest(url, "music"))
    page = await _page(browser, url)
    await _open_audio(page, "Music")
    await page.locator("#ws-audio-style").fill(_STYLE)
    await page.get_by_role("button", name="Verse", exact=True).first.click()
    await page.locator("#ws-lyrics-block-0").fill(_VERSE)
    await page.get_by_role("button", name="Chorus", exact=True).first.click()
    await page.locator("#ws-lyrics-block-1").fill(_CHORUS)
    await page.locator("#ws-tempo").fill("92")
    await _blur(page)
    await _save(page, "audio-music")


async def shot_audio_describe_it(browser: Browser, url: str) -> None:
    """Run Fill in from an idea on Music, then capture Compose with Describe it folded and the filled fields marked."""
    _order_outputs(_newest(url, "music"))
    page = await _page(browser, url)
    await _open_audio(page, "Music")
    await page.locator("#ws-describe").fill("A mellow folk song about the first snow of winter, sung softly by a woman, slow and warm")
    await page.locator("#ws-describe-fill").click()
    await page.get_by_test_id("describe-summary").wait_for(timeout=600_000)
    await page.wait_for_timeout(1500)
    await _dismiss_toasts(page)
    await _save(page, "audio-describe-it", {"x": 0, "y": 95, "width": 362, "height": 805})


async def shot_audio_script_helper(browser: Browser, url: str) -> None:
    """Capture the Speech text with the Script helper popover open on Fit to time, 30 seconds typed."""
    _order_outputs(_newest(url, "speech"))
    page = await _page(browser, url)
    await _open_audio(page, "Speech")
    await page.locator("#ws-prompt").fill(_SPEECH_LONG)
    await page.locator("#ws-script-helper-toggle").click()
    await page.wait_for_timeout(600)
    await page.get_by_role("button", name=re.compile("Fit to time")).click()
    await page.locator("#ws-script-helper-seconds").fill("30")
    await page.wait_for_timeout(900)
    await _save(page, "audio-script-helper", {"x": 0, "y": 95, "width": 900, "height": 560})


async def shot_models_audio(browser: Browser, url: str) -> None:
    """Capture the Audio models table of the Models page."""
    page = await _page(browser, f"{url}/#/models")
    await page.wait_for_timeout(2500)
    table = page.get_by_test_id("audio-models")
    await table.evaluate("element => element.scrollIntoView({block: 'center'})")
    await page.wait_for_timeout(700)
    box = await table.bounding_box()
    assert box is not None
    await _save(page, "models-audio", {"x": 110, "y": max(box["y"] - 55, 0), "width": 820, "height": box["height"] + 75})


async def shot_gallery_audio(browser: Browser, url: str) -> None:
    """Capture the Gallery filtered to Audio only."""
    page = await _page(browser, f"{url}/#/gallery")
    await page.wait_for_timeout(1500)
    await page.get_by_role("combobox", name=re.compile("media", re.IGNORECASE)).first.click()
    await page.get_by_role("option", name="Audio only").click()
    await _blur(page)
    await page.wait_for_timeout(800)
    await _save(page, "gallery-audio")


async def _open_clip_paused(page: Page, url: str, fraction: float) -> None:
    """Open the viewer on the newest video clip and pause it at *fraction* of its length, which opens the frame menu."""
    clip = _newest(url, "txt2vid")
    assert clip is not None, "the Gallery has no video clip; run the first-video shot first"
    await page.wait_for_timeout(1500)
    await page.locator(f'[aria-label="View {clip}"]').first.click(force=True)
    await page.wait_for_timeout(2000)
    box = await page.get_by_label("Playback position").bounding_box()
    assert box is not None
    await page.mouse.click(box["x"] + box["width"] * fraction, box["y"] + box["height"] / 2)
    await page.wait_for_timeout(1000)
    await page.locator('[aria-haspopup="menu"]').last.click()
    await page.wait_for_timeout(900)


async def shot_viewer_frame_menu(browser: Browser, url: str) -> None:
    """Capture a paused video with its time as a menu button and the frame menu open."""
    page = await _page(browser, f"{url}/#/gallery")
    await _open_clip_paused(page, url, 0.53)
    await _save(page, "viewer-frame-menu", {"x": 300, "y": 190, "width": 840, "height": 500})


async def shot_extend_video(browser: Browser, url: str) -> None:
    """Capture the Video tab on Extend with the clip cut at the paused frame."""
    _order_outputs(_newest(url, "txt2vid"))
    page = await _page(browser, f"{url}/#/gallery")
    await _open_clip_paused(page, url, 0.53)
    await page.get_by_role("menuitem", name=re.compile("Extend from")).click()
    await page.wait_for_timeout(2500)
    await _dismiss_toasts(page)
    await _save(page, "extend-video", {"x": 0, "y": 0, "width": 900, "height": 780})


async def shot_phone_audio(browser: Browser, url: str) -> None:
    """Capture the Audio tab on a phone, on Speech, with a speech take in the preview."""
    _order_outputs(_newest(url, "speech"))
    page = await _page(browser, f"{url}/#/workspace?workflow=speech", phone=True)
    await page.wait_for_timeout(1500)
    await _save(page, "phone-audio")


async def shot_make_media(browser: Browser, url: str) -> None:
    """Make the demo media the Gallery lacks with the app: an instrumental, an ambience, a speech take and a six-second clip."""
    if _newest(url, "music") is None:
        page = await _page(browser, url)
        await _open_audio(page, "Music", model="Stable Audio 3")
        await page.locator("#ws-audio-style").fill("Warm lo-fi piano loop with soft vinyl crackle, mellow and slow, late-night mood")
        await page.locator("#ws-submit").click()
        await _wait_job(url)
    if _newest(url, "sound") is None:
        page = await _page(browser, url)
        await _open_audio(page, "Sound")
        await page.locator("#ws-prompt").fill("Soft rain on a tin roof, distant thunder rolling, a gutter dripping")
        await page.locator("#ws-submit").click()
        await _wait_job(url)
    if _newest(url, "speech") is None:
        page = await _page(browser, url)
        await _open_audio(page, "Speech")
        await page.locator("#ws-prompt").fill(_SPEECH_TEXT)
        await page.locator("#ws-submit").click()
        await _wait_job(url)
    if _newest(url, "txt2vid") is None:
        page = await _page(browser, url)
        await _fill_video(page)
        await page.locator("#ws-submit").click()
        await _wait_job(url, timeout=3000)


async def _fill_video(page: Page) -> None:
    """Open the Video tab and write the clip's Scene and Action and its six seconds."""
    await page.get_by_role("button", name="Video", exact=True).click()
    await page.wait_for_timeout(1000)
    await page.get_by_placeholder(re.compile("Who and where")).fill("A paper boat on a rain-soaked cobblestone street at dusk, warm shop lights reflected in the puddles.")
    await page.get_by_placeholder(re.compile("What happens")).fill("The boat drifts slowly down the gutter while the camera tracks alongside, raindrops rippling the water.")
    await page.get_by_label(re.compile("Duration")).first.fill("6")
    await _blur(page)


async def _open_edit(browser: Browser, url: str) -> Page:
    """Open the hero on the Image tab's Edit kind from the viewer, with nothing painted."""
    page = await _page(browser, f"{url}/#/gallery")
    await page.locator(f'[aria-label="View {_HERO}"]').first.click(force=True)
    await page.wait_for_timeout(1200)
    await page.keyboard.press("m")
    await page.wait_for_timeout(2500)
    return page


async def _paint_moon(page: Page) -> None:
    """Open the painter and paint over the left moon in overlapping strokes, then move the pointer off the picture."""
    await page.get_by_test_id("edit-open-painter").first.click()
    await page.wait_for_timeout(1500)
    cx, cy, r = 500, 290, 140
    for y in range(cy - r + 10, cy + r - 9, 20):
        half = (r * r - (y - cy) ** 2) ** 0.5
        await page.mouse.move(cx - half, y)
        await page.mouse.down()
        await page.mouse.move(cx + half, y, steps=12)
        await page.mouse.up()
    await page.mouse.move(4, 4)


async def shot_edit_painter_paint(browser: Browser, url: str) -> None:
    """Capture the painter on Paint with the left moon painted. No job runs."""
    page = await _open_edit(browser, url)
    await _paint_moon(page)
    await _save(page, "edit-painter-paint")


async def shot_edit_painter_extend(browser: Browser, url: str) -> None:
    """Capture the painter on Extend with the right side grown. No job runs."""
    page = await _open_edit(browser, url)
    await page.get_by_test_id("edit-open-painter").first.click()
    await page.wait_for_timeout(1500)
    await page.get_by_test_id("painter-mode").get_by_role("button", name="Extend").click()
    await page.wait_for_timeout(600)
    await page.get_by_role("button", name="+ Right").click()
    await page.wait_for_timeout(800)
    await _save(page, "edit-painter-extend")


async def shot_edit_image_to_edit(browser: Browser, url: str) -> None:
    """Capture Image to edit after painting, with the Edit kind chosen and the prompt written. No job runs."""
    page = await _open_edit(browser, url)
    await _paint_moon(page)
    await page.get_by_role("button", name="Done").first.click()
    await page.wait_for_timeout(1200)
    await page.get_by_placeholder("What should be here?").fill("a ringed gas giant, pale blue, in the same evening light")
    await _blur(page)
    await _dismiss_toasts(page)
    await _save(page, "edit-image-to-edit", {"x": 0, "y": 0, "width": 1000, "height": 620})


async def shot_first_audio(browser: Browser, url: str) -> None:
    """Capture the Audio tab on Speech with the text written, the Voice in Settings and a speech take in the preview."""
    _order_outputs(_newest(url, "speech"))
    page = await _page(browser, url)
    await _open_audio(page, "Speech")
    await page.locator("#ws-prompt").fill(_SPEECH_TEXT)
    await _blur(page)
    await _save(page, "first-audio", {"x": 0, "y": 0, "width": 900, "height": 640})


async def shot_audio_take(browser: Browser, url: str) -> None:
    """Capture a finished speech take in the player."""
    _order_outputs(_newest(url, "speech"))
    page = await _page(browser, url)
    await _open_audio(page, "Speech")
    await _blur(page)
    await _save(page, "audio-take", {"x": 440, "y": 285, "width": 920, "height": 310})


async def shot_first_video(browser: Browser, url: str) -> None:
    """Capture the Video tab with the prompt written and the finished clip in the preview."""
    _order_outputs(_newest(url, "txt2vid"))
    page = await _page(browser, url)
    await _fill_video(page)
    await _dismiss_toasts(page)
    await _save(page, "first-video", {"x": 0, "y": 0, "width": 1440, "height": 650})


SHOTS: dict[str, Callable[[Browser, str], Awaitable[None]]] = {
    "make-media": shot_make_media,
    "first-audio": shot_first_audio,
    "audio-take": shot_audio_take,
    "first-video": shot_first_video,
    "workspace": shot_workspace,
    "enhance-popover": shot_enhance_popover,
    "gallery-page": shot_gallery_page,
    "viewer-details": shot_viewer_details,
    "models-page": shot_models_page,
    "config-page": shot_config_page,
    "prompts-page": shot_prompts_page,
    "preset-menu": shot_preset_menu,
    "phone-workspace": shot_phone_workspace,
    "phone-audio": shot_phone_audio,
    "audio-music": shot_audio_music,
    "audio-describe-it": shot_audio_describe_it,
    "audio-script-helper": shot_audio_script_helper,
    "models-audio": shot_models_audio,
    "gallery-audio": shot_gallery_audio,
    "viewer-frame-menu": shot_viewer_frame_menu,
    "extend-video": shot_extend_video,
    "edit-image-to-edit": shot_edit_image_to_edit,
    "edit-painter-paint": shot_edit_painter_paint,
    "edit-painter-extend": shot_edit_painter_extend,
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
        failed: list[str] = []
        runs = [(name, shot, args.url) for name, shot in SHOTS.items()]
        runs += [(name, shot, args.empty_url) for name, shot in EMPTY_SHOTS.items() if args.empty_url]
        for name, shot, base in runs:
            if wanted and name not in wanted:
                continue
            try:
                await shot(browser, base.rstrip("/"))
            except Exception as err:  # keep going: one failed shot should not lose the rest of a long session
                failed.append(name)
                print(f"FAILED {name}: {type(err).__name__}: {str(err).splitlines()[0] if str(err) else ''}")
            finally:
                _order_outputs()  # a shot may move a take to the front; restore the hero as the newest
        await browser.close()
    if failed:
        raise SystemExit(f"failed shots: {', '.join(failed)}")


if __name__ == "__main__":
    asyncio.run(main())
