"""Unit tests for the contrast and saturation adjusters."""

from __future__ import annotations

import pytest
from PIL import Image

from zvisiongenerator.processing.contrast import adjust_contrast
from zvisiongenerator.processing.saturation import adjust_saturation

_ADJUSTERS = pytest.mark.parametrize(
    ("adjust", "name"),
    [(adjust_contrast, "contrast"), (adjust_saturation, "saturation")],
    ids=["contrast", "saturation"],
)


def _make_image():
    return Image.new("RGB", (100, 100), (128, 64, 192))


@_ADJUSTERS
class TestAdjusters:
    def test_identity_at_1_0(self, adjust, name):
        img = _make_image()
        assert adjust(img, 1.0) is img  # same object returned

    def test_other_amounts_change_the_pixels(self, adjust, name):
        img = _make_image()
        result = adjust(img, 1.5)
        assert result is not img
        assert result.size == img.size
        assert result.getpixel((0, 0)) != img.getpixel((0, 0))

    def test_negative_amount_raises(self, adjust, name):
        with pytest.raises(ValueError, match=f"{name} amount must be >= 0"):
            adjust(_make_image(), -0.5)

    def test_input_not_mutated(self, adjust, name):
        img = _make_image()
        original_pixel = img.getpixel((0, 0))
        adjust(img, 1.5)
        assert img.getpixel((0, 0)) == original_pixel

    def test_zero_amount_produces_grey(self, adjust, name):
        img = _make_image()
        result = adjust(img, 0.0)
        assert result is not img
        assert result.size == img.size
        r, g, b = result.getpixel((0, 0))
        assert r == g == b
