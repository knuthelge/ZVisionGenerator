"""End-to-end review before upscaling through run_batch, the real workflow and a mocked backend."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from tests._factories import _make_args, _make_plan

from zvisiongenerator.image_runner import run_batch
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.interactive import SkipSignal

_MODEL_INFO = ImageModelInfo(family="zimage", is_distilled=False, size=None)
_CONFIG = {
    "sizes": {"2:3": {"m": {"width": 512, "height": 512}}},
    "generation": {"seed_min": 1, "seed_max": 100},
    "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4},
    "model_presets": {"zimage": {"supports_negative_prompt": True}},
}
_PROMPTS = {"set1": [("a photo of a cat", None)]}


_SMALL_RENDER = Image.effect_noise((64, 64), 40).convert("RGB")  # the same pixels in every run, so outputs can be compared


def _backend() -> MagicMock:
    backend = MagicMock()
    backend.name = "mock"
    backend.text_to_image.side_effect = lambda *a, **kw: _SMALL_RENDER.copy()
    backend.image_to_image.side_effect = lambda *a, **kw: Image.new("RGB", (128, 128), color=(10, 20, 30))
    return backend


def _run(out: Path, *, review: bool, skip: SkipSignal | None = None, progress_callback=None) -> MagicMock:
    backend = _backend()
    args = _make_args(upscale=2, upscale_denoise=0.3, upscale_steps=2, review=review, output=str(out), sharpen=False, width=64, height=64)
    run_batch(
        backend,
        MagicMock(spec=[]),
        _PROMPTS,
        _CONFIG,
        args,
        model_info=_MODEL_INFO,
        plan=_make_plan(_PROMPTS, args, config=_CONFIG),
        skip_signal=skip,
        progress_callback=progress_callback,
    )
    return backend


def _png_pixels(folder: Path) -> list[bytes]:
    return [Image.open(path).convert("RGB").tobytes() for path in sorted(folder.glob("*.png"))]


def test_an_approved_review_gives_the_same_output_and_upscale_input_as_no_review(tmp_path):
    plain = _run(tmp_path / "plain", review=False)

    skip = SkipSignal()
    reviewed_events: list[dict] = []

    def _on_event(event):
        reviewed_events.append(event)
        if event["type"] == "job_paused" and "review" in event:
            skip.set_review_decision("approve")

    reviewed = _run(tmp_path / "reviewed", review=True, skip=skip, progress_callback=_on_event)

    assert any(e["type"] == "job_paused" and "review" in e for e in reviewed_events)

    def _images_sent(call):
        return [v.tobytes() for v in (*call.args, *call.kwargs.values()) if isinstance(v, Image.Image)]

    assert plain.image_to_image.call_count == reviewed.image_to_image.call_count == 1
    assert _images_sent(plain.image_to_image.call_args) == _images_sent(reviewed.image_to_image.call_args) != []
    assert _png_pixels(tmp_path / "plain") == _png_pixels(tmp_path / "reviewed")
    assert len(_png_pixels(tmp_path / "reviewed")) == 1


def test_a_discard_writes_no_file_and_never_upscales(tmp_path):
    skip = SkipSignal()

    def _on_event(event):
        if event["type"] == "job_paused" and "review" in event:
            skip.set_review_decision("discard")

    backend = _run(tmp_path / "out", review=True, skip=skip, progress_callback=_on_event)

    backend.image_to_image.assert_not_called()
    assert _png_pixels(tmp_path / "out") == []


def test_review_without_a_signal_is_refused(tmp_path):
    with pytest.raises(ValueError, match="skip_signal"):
        _run(tmp_path / "out", review=True, skip=None)
