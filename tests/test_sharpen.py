"""Tests for zvisiongenerator.processing.sharpen — contrast_adaptive_sharpening."""

from __future__ import annotations

import numpy as np
from PIL import Image

from zvisiongenerator.processing.sharpen import contrast_adaptive_sharpening


class TestContrastAdaptiveSharpening:
    def test_amount_zero_returns_unchanged(self):
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        result = contrast_adaptive_sharpening(img, amount=0.0)
        assert result.mode == img.mode
        assert result.size == img.size
        assert result.getpixel((0, 0)) == img.getpixel((0, 0))
        assert result.tobytes() == img.tobytes()

    def test_negative_amount_returns_unchanged(self):
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        result = contrast_adaptive_sharpening(img, amount=-0.5)
        assert result.mode == img.mode
        assert result.size == img.size
        assert result.getpixel((0, 0)) == img.getpixel((0, 0))
        assert result.tobytes() == img.tobytes()

    def test_positive_amount_modifies_pixels(self):
        # Create an image with some variation for sharpening to act on
        arr = np.random.RandomState(42).randint(50, 200, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = contrast_adaptive_sharpening(img, amount=0.8)
        result_arr = np.asarray(result)
        # At least some pixels should differ
        assert not np.array_equal(np.asarray(img), result_arr)

    def test_output_same_dimensions(self):
        img = Image.new("RGB", (80, 60), color=(200, 100, 50))
        result = contrast_adaptive_sharpening(img, amount=0.5)
        assert result.size == (80, 60)

    def test_output_is_rgb(self):
        img = Image.new("RGB", (32, 32), color=(0, 255, 0))
        result = contrast_adaptive_sharpening(img, amount=0.5)
        assert result.mode == "RGB"

    def test_uniform_image_stays_similar(self):
        # A perfectly uniform image should not change much
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        result = contrast_adaptive_sharpening(img, amount=1.0)
        result_arr = np.asarray(result)
        original_arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        # All values should be within ±1 of original due to floating point
        assert np.allclose(result_arr, original_arr, atol=1)

    def test_max_amount_produces_valid_output(self):
        arr = np.random.RandomState(99).randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = contrast_adaptive_sharpening(img, amount=1.0)
        result_arr = np.asarray(result)
        assert result_arr.min() >= 0
        assert result_arr.max() <= 255

    def test_converts_non_rgb_input(self):
        # RGBA should be converted to RGB internally
        img = Image.new("RGBA", (32, 32), color=(128, 128, 128, 255))
        result = contrast_adaptive_sharpening(img, amount=0.5)
        assert result.mode == "RGB"
