"""Pure helpers for cutting a subject out of an image: trimap, working size, alpha merge, foreground colours, RGBA.

The helpers work on NumPy arrays and PIL images, and on torch tensors for the foreground estimate. They load no
model, touch no accelerator-specific API and import neither OpenCV nor pymatting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from zvisiongenerator.utils.input_images import flatten_alpha

if TYPE_CHECKING:
    from torch import Tensor

MATTE_BACKDROP = (200, 200, 200)
"""Light grey an image is flattened onto when a model needs an opaque input (the transparent upscale)."""

TRIMAP_RADIUS = 12
"""Half-width, in pixels at the 1024 px working size, of the unknown band the matting model refines."""

MASK_SIZE = 1024
"""Side of the square the segmentation model runs at."""

VITMATTE_MAX = 1024
"""Longest side the matting model runs at; its global attention needs too much memory above it."""

_SURE_FOREGROUND = 0.95
_SURE_BACKGROUND = 0.05
_UNKNOWN = 0.5


def trimap(mask: np.ndarray, *, radius: int) -> np.ndarray:
    """Return a trimap (0 background, 0.5 unknown, 1 foreground) with an unknown band of *radius* around the edge.

    Sure foreground is ``mask > 0.95`` and sure background is ``mask < 0.05``, each eroded by a disk of *radius*.
    Pixels the mask leaves soft (between 0.05 and 0.95) stay unknown, so hair the mask half-kept is left for the
    matting model. The image border does not erode: pixels outside the image count as sure.

    Args:
        mask: Segmentation mask as a float ``(H, W)`` array in [0, 1].
        radius: Radius of the erosion disk in pixels.

    Returns:
        A float32 ``(H, W)`` array holding only 0, 0.5 and 1.
    """
    from scipy.ndimage import binary_erosion

    disk = _disk(radius)
    sure_foreground = binary_erosion(mask > _SURE_FOREGROUND, structure=disk, border_value=1)
    sure_background = binary_erosion(mask < _SURE_BACKGROUND, structure=disk, border_value=1)
    result = np.full(mask.shape, _UNKNOWN, dtype=np.float32)
    result[sure_foreground] = 1.0
    result[sure_background] = 0.0
    return result


def working_size(width: int, height: int, *, longest: int = MASK_SIZE) -> tuple[int, int]:
    """Return ``(width, height)`` scaled to fit *longest* on the longer side, keeping the aspect ratio and never upscaling."""
    scale = min(1.0, longest / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def merge_alpha(tri: np.ndarray, refined: np.ndarray) -> np.ndarray:
    """Return *refined* where *tri* is unknown and *tri* itself elsewhere, clipped to [0, 1].

    The matting model only decides the unknown band; the sure regions of the trimap stay exactly 0 or 1.
    """
    return np.clip(np.where(tri == _UNKNOWN, refined, tri), 0.0, 1.0)


def on_backdrop(image: Image.Image, colour: tuple[int, int, int] = MATTE_BACKDROP) -> Image.Image:
    """Return *image* as RGB with its transparency composited over *colour*."""
    return flatten_alpha(image, background=colour)


def estimate_foreground(
    rgb: Tensor,
    alpha: Tensor,
    *,
    regularization: float = 1e-5,
    n_small_iterations: int = 10,
    n_big_iterations: int = 2,
    small_size: int = 32,
    gradient_weight: float = 1.0,
) -> Tensor:
    """Return the subject colours under *alpha*, with the background's colour taken out of the edges.

    A pixel on a soft edge shows ``alpha * F + (1 - alpha) * B``, so keeping its colour as the foreground leaves a
    halo of the old background. This solves for ``F`` and ``B`` together with the multilevel method of Germer et al.,
    "Fast Multi-Level Foreground Estimation" (2020): start from one pixel, double the size level by level, and at
    each level run a few sweeps in which every pixel solves a 2 x 2 system tying its colour to its four neighbours,
    weighted by how much the alpha changes towards them.

    This is a port of ``estimate_foreground_ml`` from pymatting (MIT licence, Copyright (c) 2020 PyMatting)
    to torch. It runs on any device. Pymatting sweeps pixels one after another, so each sees its
    neighbours' fresh values; here a sweep updates all pixels at once from the previous sweep's values, which is
    what a vectorised kernel can do. The levels do the long-range work, so the result stays close.

    Args:
        rgb: Image as a float ``(H, W, 3)`` tensor in [0, 1].
        alpha: Alpha matte as a float ``(H, W)`` tensor in [0, 1], on the same device.
        regularization: Smoothness strength; higher gives smoother colours.
        n_small_iterations: Sweeps per level while both sides are at most *small_size*.
        n_big_iterations: Sweeps per level above that.
        small_size: Side below which a level counts as small.
        gradient_weight: Extra weight of an alpha change between neighbours; higher gives smoother foregrounds.

    Returns:
        The foreground as a float32 ``(H, W, 3)`` tensor in [0, 1] on the input's device.
    """
    import torch

    with torch.no_grad():
        image = rgb.to(torch.float32)
        matte = alpha.to(torch.float32)
        height, width, depth = image.shape
        foreground = _mean_colour(image, matte > 0.9).reshape(1, 1, depth)
        background = _mean_colour(image, matte < 0.1).reshape(1, 1, depth)
        levels = int(np.ceil(np.log2(max(width, height))))
        for level in range(levels + 1):
            size = (max(1, round(height ** (level / levels))), max(1, round(width ** (level / levels)))) if levels else (height, width)
            level_image, level_alpha = _nearest(image, size), _nearest(matte, size)
            foreground, background = _nearest(foreground, size), _nearest(background, size)
            small = size[0] <= small_size and size[1] <= small_size
            for _ in range(n_small_iterations if small else n_big_iterations):
                foreground, background = _sweep(level_image, level_alpha, foreground, background, regularization, gradient_weight)
        return foreground


def to_rgba(foreground: np.ndarray, alpha: np.ndarray) -> Image.Image:
    """Return an 8-bit RGBA image from float *foreground* ``(H, W, 3)`` and *alpha* ``(H, W)`` arrays in [0, 1]."""
    rgba = np.concatenate([np.clip(foreground, 0.0, 1.0), np.clip(alpha, 0.0, 1.0)[..., None]], axis=-1)
    return Image.fromarray((rgba * 255.0).round().astype(np.uint8), mode="RGBA")


def _disk(radius: int) -> np.ndarray:
    """Return a boolean ``(2r + 1, 2r + 1)`` disk structuring element."""
    offsets = np.arange(-radius, radius + 1)
    return offsets[:, None] ** 2 + offsets[None, :] ** 2 <= radius**2


def _mean_colour(image: Tensor, selected: Tensor) -> Tensor:
    """Return the mean colour of the *selected* pixels, or zeros when there are none (pymatting's ``count + 1e-5``)."""
    return image[selected].sum(dim=0) / (selected.sum() + 1e-5)


def _nearest(values: Tensor, size: tuple[int, int]) -> Tensor:
    """Return *values* (``(H, W)`` or ``(H, W, C)``) resized to *size* by nearest neighbour, as pymatting indexes."""
    import torch

    height, width = values.shape[:2]
    rows = (torch.arange(size[0], device=values.device) * height // size[0]).clamp(0, height - 1)
    columns = (torch.arange(size[1], device=values.device) * width // size[1]).clamp(0, width - 1)
    return values[rows][:, columns]


def _neighbours(values: Tensor) -> list[Tensor]:
    """Return *values* shifted so each pixel sees its left, right, upper and lower neighbour; the border sees itself."""
    import torch.nn.functional as functional

    planes = values.unsqueeze(-1) if values.ndim == 2 else values
    padded = functional.pad(planes.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="replicate")[0].permute(1, 2, 0)
    height, width = planes.shape[:2]
    shifted = [padded[1 : 1 + height, 0:width], padded[1 : 1 + height, 2 : 2 + width], padded[0:height, 1 : 1 + width], padded[2 : 2 + height, 1 : 1 + width]]
    return [plane.squeeze(-1) for plane in shifted] if values.ndim == 2 else shifted


def _sweep(image: Tensor, alpha: Tensor, foreground: Tensor, background: Tensor, regularization: float, gradient_weight: float) -> tuple[Tensor, Tensor]:
    """Return one sweep's ``(foreground, background)``: every pixel solves its 2 x 2 system from the previous values."""
    a0 = alpha
    a1 = 1.0 - alpha
    a00, a01, a11 = a0 * a0, a0 * a1, a1 * a1
    b0, b1 = a0.unsqueeze(-1) * image, a1.unsqueeze(-1) * image
    for neighbour_alpha, neighbour_foreground, neighbour_background in zip(_neighbours(alpha), _neighbours(foreground), _neighbours(background), strict=True):
        weight = regularization + gradient_weight * (a0 - neighbour_alpha).abs()
        a00, a11 = a00 + weight, a11 + weight
        b0 = b0 + weight.unsqueeze(-1) * neighbour_foreground
        b1 = b1 + weight.unsqueeze(-1) * neighbour_background
    inverse = (1.0 / (a00 * a11 - a01 * a01)).unsqueeze(-1)
    b00, b01, b11 = inverse * a11.unsqueeze(-1), -inverse * a01.unsqueeze(-1), inverse * a00.unsqueeze(-1)
    return (b00 * b0 + b01 * b1).clamp(0.0, 1.0), (b01 * b0 + b11 * b1).clamp(0.0, 1.0)
