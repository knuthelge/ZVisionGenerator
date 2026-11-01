"""Tests for the matte helpers: trimap, working size, alpha merge, backdrop, foreground estimate and RGBA assembly."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from zvisiongenerator.processing.matte import (
    MASK_SIZE,
    MATTE_BACKDROP,
    TRIMAP_RADIUS,
    VITMATTE_MAX,
    estimate_foreground,
    merge_alpha,
    on_backdrop,
    to_rgba,
    trimap,
    working_size,
)

_RED = np.array([1.0, 0.0, 0.0], dtype=np.float32)
_GREY = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _disk_mask(size: int, radius: float) -> np.ndarray:
    """Return a hard 0/1 mask with a centred disk of *radius*."""
    rows, columns = np.mgrid[:size, :size]
    return (((rows - size // 2) ** 2 + (columns - size // 2) ** 2) <= radius**2).astype(np.float32)


def _red_on_grey(*, size: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(image, alpha)``: a red left half, a half-alpha band, then clear, every pixel mixed over grey."""
    alpha = np.zeros((size, size), dtype=np.float32)
    alpha[:, : size // 2 - 2] = 1.0
    alpha[:, size // 2 - 2 : size // 2 + 2] = 0.5
    image = alpha[..., None] * _RED + (1.0 - alpha[..., None]) * _GREY
    return image.astype(np.float32), alpha


def _reference_foreground(image: np.ndarray, alpha: np.ndarray, *, regularization: float, n_small: int, n_big: int, small_size: int, gradient_weight: float) -> np.ndarray:
    """Return the multilevel foreground estimate by explicit per-pixel loops, with every sweep reading the previous one."""
    height, width, depth = image.shape
    fg_mean = image[alpha > 0.9].sum(axis=0) / ((alpha > 0.9).sum() + 1e-5)
    bg_mean = image[alpha < 0.1].sum(axis=0) / ((alpha < 0.1).sum() + 1e-5)
    foreground, background = fg_mean.reshape(1, 1, depth), bg_mean.reshape(1, 1, depth)
    levels = int(np.ceil(np.log2(max(width, height))))
    for level in range(levels + 1):
        h, w = round(height ** (level / levels)), round(width ** (level / levels))
        rows, columns = [y * height // h for y in range(h)], [x * width // w for x in range(w)]
        level_image, level_alpha = image[rows][:, columns], alpha[rows][:, columns]
        prev_h, prev_w = foreground.shape[:2]
        prev_rows, prev_columns = [y * prev_h // h for y in range(h)], [x * prev_w // w for x in range(w)]
        foreground, background = foreground[prev_rows][:, prev_columns], background[prev_rows][:, prev_columns]
        for _ in range(n_small if h <= small_size and w <= small_size else n_big):
            new_f, new_b = np.empty_like(foreground), np.empty_like(background)
            for y in range(h):
                for x in range(w):
                    a0 = float(level_alpha[y, x])
                    a1 = 1.0 - a0
                    a00, a01, a11 = a0 * a0, a0 * a1, a1 * a1
                    b0, b1 = a0 * level_image[y, x].astype(np.float64), a1 * level_image[y, x].astype(np.float64)
                    for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                        y2, x2 = min(max(y + dy, 0), h - 1), min(max(x + dx, 0), w - 1)
                        weight = regularization + gradient_weight * abs(a0 - float(level_alpha[y2, x2]))
                        a00, a11 = a00 + weight, a11 + weight
                        b0 = b0 + weight * foreground[y2, x2]
                        b1 = b1 + weight * background[y2, x2]
                    solution = np.linalg.solve(np.array([[a00, a01], [a01, a11]]), np.stack([b0, b1]))
                    new_f[y, x], new_b[y, x] = np.clip(solution, 0.0, 1.0)
            foreground, background = new_f, new_b
    return foreground


class TestTrimap:
    def test_the_unknown_band_is_as_wide_as_the_radius(self) -> None:
        mask = _disk_mask(129, 40)
        result = trimap(mask, radius=8)

        assert set(np.unique(result)) == {0.0, 0.5, 1.0}
        row = result[64]
        foreground_edge = np.flatnonzero(row == 1.0)
        # Sure foreground ends 8 px inside the disk's edge at column 64 + 40, and sure background starts 8 px outside.
        assert foreground_edge.max() == 64 + 40 - 8
        assert np.flatnonzero(row == 0.0)[np.flatnonzero(row == 0.0) > 64].min() == 64 + 40 + 8 + 1

    def test_soft_pixels_stay_unknown(self) -> None:
        mask = np.zeros((40, 40), dtype=np.float32)
        mask[:, :20] = 1.0
        mask[:, 20:] = 0.5
        result = trimap(mask, radius=2)

        assert (result[:, 22:] == 0.5).all()
        assert result[20, 0] == 1.0

    def test_the_image_border_does_not_erode(self) -> None:
        result = trimap(np.ones((30, 30), dtype=np.float32), radius=5)

        assert (result == 1.0).all()
        assert (trimap(np.zeros((30, 30), dtype=np.float32), radius=5) == 0.0).all()

    def test_a_zero_radius_keeps_every_decided_pixel(self) -> None:
        mask = _disk_mask(31, 10)

        np.testing.assert_array_equal(trimap(mask, radius=0), mask)


class TestWorkingSize:
    def test_a_large_image_is_scaled_to_the_longest_side(self) -> None:
        assert working_size(2048, 1024) == (1024, 512)
        assert working_size(1000, 3000, longest=1500) == (500, 1500)

    def test_a_small_image_is_never_upscaled(self) -> None:
        assert working_size(800, 600) == (800, 600)

    def test_a_sliver_keeps_at_least_one_pixel(self) -> None:
        assert working_size(10000, 3) == (1024, 1)

    def test_the_constants_match_the_working_size(self) -> None:
        assert (MASK_SIZE, VITMATTE_MAX, TRIMAP_RADIUS) == (1024, 1024, 12)


class TestMergeAlpha:
    def test_sure_regions_are_kept(self) -> None:
        tri = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        refined = np.array([[0.7, 0.3, 0.2]], dtype=np.float32)

        np.testing.assert_allclose(merge_alpha(tri, refined), [[0.0, 0.3, 1.0]])

    def test_the_unknown_band_takes_the_refined_alpha_clipped(self) -> None:
        tri = np.full((1, 2), 0.5, dtype=np.float32)

        np.testing.assert_allclose(merge_alpha(tri, np.array([[-0.2, 1.4]], dtype=np.float32)), [[0.0, 1.0]])


class TestOnBackdrop:
    def test_a_half_transparent_pixel_mixes_half_and_half(self) -> None:
        image = Image.new("RGBA", (2, 2), (255, 0, 0, 128))
        result = on_backdrop(image, (0, 0, 200))

        assert result.mode == "RGB"
        red, green, blue = result.getpixel((0, 0))
        assert (red, green, blue) == (pytest.approx(128, abs=1), 0, pytest.approx(100, abs=1))

    def test_the_default_is_the_light_grey_backdrop(self) -> None:
        clear = Image.new("RGBA", (2, 2), (255, 0, 0, 0))

        assert on_backdrop(clear).getpixel((1, 1)) == MATTE_BACKDROP

    def test_an_opaque_image_is_returned_as_rgb(self) -> None:
        assert on_backdrop(Image.new("L", (2, 2), 7)).getpixel((0, 0)) == (7, 7, 7)


class TestEstimateForeground:
    def test_it_matches_a_numpy_reference(self) -> None:
        rng = np.random.default_rng(3)
        alpha = np.clip(np.cumsum(rng.normal(0.0, 0.3, size=(9, 12)), axis=1) * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
        image = (alpha[..., None] * rng.random((9, 12, 3)) + (1.0 - alpha[..., None]) * rng.random((1, 1, 3))).astype(np.float32)
        options = {"regularization": 1e-5, "n_small": 4, "n_big": 2, "small_size": 6, "gradient_weight": 1.0}

        expected = _reference_foreground(image, alpha, **options)
        got = estimate_foreground(
            torch.from_numpy(image),
            torch.from_numpy(alpha),
            regularization=1e-5,
            n_small_iterations=4,
            n_big_iterations=2,
            small_size=6,
        )

        assert got.dtype == torch.float32
        assert got.shape == (9, 12, 3)
        np.testing.assert_allclose(got.numpy(), expected, atol=1e-3)

    def test_a_half_alpha_edge_returns_the_subject_colour(self) -> None:
        image, alpha = _red_on_grey()
        foreground = estimate_foreground(torch.from_numpy(image), torch.from_numpy(alpha)).numpy()
        band = foreground[:, 22:26]

        np.testing.assert_allclose(band, np.broadcast_to(_RED, band.shape), atol=0.05)
        # Keeping the pixels as they are leaves the edge tinted by the grey it was mixed with.
        assert np.abs(image[:, 22:26] - _RED).max() > 0.2

    def test_the_result_stays_in_range_and_on_the_input_device(self) -> None:
        image, alpha = _red_on_grey(size=20)
        foreground = estimate_foreground(torch.from_numpy(image), torch.from_numpy(alpha))

        assert foreground.device.type == "cpu"
        assert 0.0 <= float(foreground.min()) and float(foreground.max()) <= 1.0

    def test_a_single_pixel_image_is_returned_clipped(self) -> None:
        foreground = estimate_foreground(torch.full((1, 1, 3), 0.4), torch.ones((1, 1)))

        assert foreground.shape == (1, 1, 3)
        np.testing.assert_allclose(foreground.numpy(), 0.4, atol=1e-3)


class TestToRgba:
    def test_the_alpha_round_trips(self) -> None:
        alpha = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
        result = to_rgba(np.full((3, 4, 3), 0.5, dtype=np.float32), alpha)

        assert result.mode == "RGBA"
        assert result.size == (4, 3)
        np.testing.assert_array_equal(np.asarray(result)[..., 3], (alpha * 255).round().astype(np.uint8))

    def test_the_colours_are_eight_bit(self) -> None:
        result = to_rgba(np.broadcast_to(_RED, (2, 2, 3)), np.ones((2, 2), dtype=np.float32))

        assert result.getpixel((0, 0)) == (255, 0, 0, 255)

    def test_out_of_range_values_are_clipped(self) -> None:
        result = to_rgba(np.full((1, 1, 3), 1.5, dtype=np.float32), np.full((1, 1), -0.5, dtype=np.float32))

        assert result.getpixel((0, 0)) == (255, 255, 255, 0)


def test_the_module_imports_neither_opencv_nor_pymatting() -> None:
    code = "import sys\nimport zvisiongenerator.processing.matte\nraise SystemExit(any(name.split('.')[0] in ('cv2', 'pymatting') for name in sys.modules))"

    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
