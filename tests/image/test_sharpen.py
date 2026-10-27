"""Tests for zvisiongenerator.processing.sharpen — contrast_adaptive_sharpening."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from zvisiongenerator.processing.sharpen import contrast_adaptive_sharpening


class TestContrastAdaptiveSharpening:
    @pytest.mark.parametrize("amount", [0.0, -0.5])
    def test_amount_zero_or_negative_returns_unchanged(self, amount):
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        result = contrast_adaptive_sharpening(img, amount=amount)
        assert result.mode == img.mode
        assert result.size == img.size
        assert result.getpixel((0, 0)) == img.getpixel((0, 0))
        assert result.tobytes() == img.tobytes()

    def test_amount_above_the_cap_is_rejected(self):
        img = Image.new("RGB", (8, 8), color=(100, 100, 100))
        with pytest.raises(ValueError, match="at most 1.5"):
            contrast_adaptive_sharpening(img, amount=1.6)

    def test_amount_at_the_cap_stays_finite(self):
        # A flat image is the worst case for the filter's normaliser.
        img = Image.new("RGB", (8, 8), color=(100, 100, 100))
        assert contrast_adaptive_sharpening(img, amount=1.5).tobytes() == img.tobytes()

    def test_positive_amount_modifies_pixels(self):
        # Create an image with some variation for sharpening to act on
        arr = np.random.RandomState(42).randint(50, 200, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = contrast_adaptive_sharpening(img, amount=0.8)
        result_arr = np.asarray(result)
        # At least some pixels should differ
        assert not np.array_equal(np.asarray(img), result_arr)

    def test_uniform_image_stays_similar(self):
        # A perfectly uniform image should not change much
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        result = contrast_adaptive_sharpening(img, amount=1.0)
        result_arr = np.asarray(result)
        original_arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        # All values should be within ±1 of original due to floating point
        assert np.allclose(result_arr, original_arr, atol=1)

    def test_converts_non_rgb_input(self):
        # RGBA should be converted to RGB internally
        img = Image.new("RGBA", (80, 60), color=(128, 128, 128, 255))
        result = contrast_adaptive_sharpening(img, amount=0.5)
        assert result.mode == "RGB"
        assert result.size == (80, 60)
