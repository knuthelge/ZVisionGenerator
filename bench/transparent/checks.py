"""Checks on a matte that need no ground truth: softness, coverage, stray specks and background bleed at the edge."""

from __future__ import annotations

import numpy as np

_SPECK_MIN_PIXELS = 50


def matte_checks(alpha: np.ndarray, foreground: np.ndarray, background: np.ndarray) -> dict[str, float]:
    """Return the checks for one matte.

    Args:
        alpha: Alpha in [0, 1], shape ``(H, W)``.
        foreground: The RGB stored under the alpha, in [0, 1].
        background: The render's background colour (RGB in [0, 1]).

    Returns:
        ``coverage`` (share of pixels over half opaque), ``soft`` (share between 2 % and 98 %), ``specks`` (opaque
        islands of at least 50 pixels apart from the largest), ``bleed`` (mean closeness of edge pixels to the
        background colour, 0 = none, 1 = the edge is background) and ``clear_on_border`` (share of the border that is
        under 10 % opaque; a good matte of an isolated subject is near 1).
    """
    import cv2

    opaque = (alpha > 0.5).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(opaque, connectivity=8)
    areas = sorted((int(area) for area in stats[1:, cv2.CC_STAT_AREA]), reverse=True)
    specks = sum(1 for area in areas[1:] if area >= _SPECK_MIN_PIXELS)
    edge = (alpha > 0.05) & (alpha < 0.95)
    distance = np.linalg.norm(foreground - background, axis=-1) / np.sqrt(3.0)
    bleed = float(1.0 - distance[edge].mean()) if edge.any() else 0.0
    border = np.concatenate([alpha[:4].ravel(), alpha[-4:].ravel(), alpha[:, :4].ravel(), alpha[:, -4:].ravel()])
    return {
        "coverage": float(opaque.mean()),
        "soft": float(((alpha > 0.02) & (alpha < 0.98)).mean()),
        "specks": float(specks),
        "bleed": bleed,
        "clear_on_border": float((border < 0.1).mean()),
    }
