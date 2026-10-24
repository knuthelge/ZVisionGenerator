"""Tests for the image-to-video conditioning-image CRF helper."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from zvisiongenerator.backends.cuda.video.image_crf import recompress

_TARGET = "diffusers.pipelines.ltx2.utils.apply_image_conditioning_crf"


def _gradient(width: int, height: int, mode: str = "RGB") -> Image.Image:
    return Image.new("RGB", (width, height), (10, 120, 200)).convert(mode)


def test_crf_zero_returns_the_same_image_without_calling_diffusers():
    image = _gradient(8, 8)
    with patch(_TARGET) as fn:
        assert recompress(image, 0) is image
    fn.assert_not_called()


@pytest.mark.parametrize("crf", [-1, 52, 100, True, 18.0, "18", None])
def test_invalid_crf_raises_value_error(crf):
    with patch(_TARGET) as fn, pytest.raises(ValueError, match="crf"):
        recompress(_gradient(8, 8), crf)
    fn.assert_not_called()


@pytest.mark.parametrize("crf", [0, 51])
def test_range_bounds_are_accepted(crf):
    with patch(_TARGET, side_effect=lambda arr, _crf: arr):
        assert isinstance(recompress(_gradient(8, 8), crf), Image.Image)


def test_diffusers_receives_uint8_rgb_array_and_crf():
    seen = {}

    def fake(arr, crf):
        seen["arr"], seen["crf"] = arr, crf
        return np.zeros_like(arr)

    with patch(_TARGET, side_effect=fake):
        out = recompress(_gradient(10, 6, "L"), 18)

    assert seen["crf"] == 18
    assert seen["arr"].dtype == np.uint8
    assert seen["arr"].shape == (6, 10, 3)
    assert out.mode == "RGB"
    assert out.size == (10, 6)


def test_non_rgb_input_is_converted_before_encoding():
    seen = {}

    def fake(arr, crf):
        seen["shape"] = arr.shape
        return arr

    with patch(_TARGET, side_effect=fake):
        recompress(Image.new("RGBA", (4, 4), (1, 2, 3, 128)), 18)

    assert seen["shape"] == (4, 4, 3)


def test_result_is_built_from_the_diffusers_output():
    result = np.full((4, 6, 3), 77, dtype=np.uint8)
    with patch(_TARGET, return_value=result):
        out = recompress(_gradient(6, 4), 18)

    assert out.size == (6, 4)
    assert np.array_equal(np.asarray(out), result)


def test_odd_size_follows_diffusers_even_crop():
    def crop_even(arr, _crf):
        return arr[: arr.shape[0] // 2 * 2, : arr.shape[1] // 2 * 2]

    with patch(_TARGET, side_effect=crop_even):
        out = recompress(_gradient(9, 7), 18)

    assert out.size == (8, 6)


def test_real_encode_handles_an_odd_size_when_av_is_available():
    pytest.importorskip("av")
    out = recompress(_gradient(33, 21), 18)

    assert out.mode == "RGB"
    assert out.size == (32, 20)
