"""Tests for shrinking a preview image to the picture kept with a downloaded LoRA."""

from __future__ import annotations

import io

from PIL import Image

from zvisiongenerator.utils import lora_preview
from zvisiongenerator.utils.lora_preview import PREVIEW_EDGE, make_preview


def _png(size: tuple[int, int], mode: str = "RGB") -> bytes:
    out = io.BytesIO()
    Image.new(mode, size, (200, 30, 30, 120) if mode == "RGBA" else (200, 30, 30)).save(out, format="PNG")
    return out.getvalue()


def test_a_large_image_is_shrunk_to_the_edge_and_kept_as_webp():
    content_type, data = make_preview(_png((1200, 800)))

    assert content_type == "image/webp"
    with Image.open(io.BytesIO(data)) as image:
        assert (image.format, max(image.size)) == ("WEBP", PREVIEW_EDGE)
        assert image.size == (PREVIEW_EDGE, round(800 * PREVIEW_EDGE / 1200))


def test_a_small_image_is_not_enlarged_and_keeps_its_transparency():
    _, data = make_preview(_png((40, 30), "RGBA"))

    with Image.open(io.BytesIO(data)) as image:
        assert (image.size, "A" in image.getbands()) == ((40, 30), True)


def test_bytes_that_are_not_an_image_give_nothing():
    assert make_preview(b"not an image") is None


def test_an_image_with_too_many_pixels_is_refused_before_it_is_decoded(monkeypatch):
    monkeypatch.setattr(lora_preview, "MAX_PREVIEW_PIXELS", 100)

    assert make_preview(_png((20, 20))) is None
