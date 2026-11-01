"""Image processing utilities — contrast, saturation, sharpening, and the helpers of the background cut-out."""

from __future__ import annotations

from zvisiongenerator.processing.contrast import adjust_contrast
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
from zvisiongenerator.processing.saturation import adjust_saturation
from zvisiongenerator.processing.sharpen import contrast_adaptive_sharpening

__all__ = [
    "MASK_SIZE",
    "MATTE_BACKDROP",
    "TRIMAP_RADIUS",
    "VITMATTE_MAX",
    "adjust_contrast",
    "adjust_saturation",
    "contrast_adaptive_sharpening",
    "estimate_foreground",
    "merge_alpha",
    "on_backdrop",
    "to_rgba",
    "trimap",
    "working_size",
]
