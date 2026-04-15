"""Unit tests for adjust_saturation."""

from __future__ import annotations

import pytest
from PIL import Image

from zvisiongenerator.processing.saturation import adjust_saturation


def _make_image():
    return Image.new("RGB", (100, 100), (128, 64, 192))


class TestAdjustSaturation:
    def test_identity_at_1_0(self):
        img = _make_image()
        result = adjust_saturation(img, 1.0)
        assert result is img  # same object returned

    def test_enhancement_at_1_5(self):
        img = _make_image()
        result = adjust_saturation(img, 1.5)
        assert result is not img
        assert result.size == img.size
        assert result.getpixel((0, 0)) != img.getpixel((0, 0))

    def test_reduction_at_0_5(self):
        img = _make_image()
        result = adjust_saturation(img, 0.5)
        assert result is not img
        assert result.size == img.size
        assert result.getpixel((0, 0)) != img.getpixel((0, 0))

    def test_negative_amount_raises(self):
        img = _make_image()
        with pytest.raises(ValueError, match="saturation amount must be >= 0"):
            adjust_saturation(img, -0.5)

    def test_input_not_mutated(self):
        img = _make_image()
        original_pixel = img.getpixel((0, 0))
        adjust_saturation(img, 1.5)
        assert img.getpixel((0, 0)) == original_pixel

    def test_zero_amount_produces_grayscale(self):
        img = _make_image()
        result = adjust_saturation(img, 0.0)
        assert result is not img
        assert result.size == img.size
        # 0.0 saturation produces a grayscale image (R == G == B for each pixel)
        r, g, b = result.getpixel((0, 0))
        assert r == g == b
