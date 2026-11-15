"""Image processing utilities — contrast, saturation, sharpening, the Adjust editor's light maths and framing, and the helpers of the background cut-out."""

from __future__ import annotations

from zvisiongenerator.processing.adjust import LightSettings, adjust_light
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
from zvisiongenerator.processing.reframe import CropBox, Reframe, framed_size, reframe
from zvisiongenerator.processing.saturation import adjust_saturation
from zvisiongenerator.processing.sharpen import contrast_adaptive_sharpening

__all__ = [
    "MASK_SIZE",
    "MATTE_BACKDROP",
    "TRIMAP_RADIUS",
    "VITMATTE_MAX",
    "CropBox",
    "LightSettings",
    "Reframe",
    "adjust_contrast",
    "adjust_light",
    "adjust_saturation",
    "contrast_adaptive_sharpening",
    "estimate_foreground",
    "framed_size",
    "merge_alpha",
    "on_backdrop",
    "reframe",
    "to_rgba",
    "trimap",
    "working_size",
]
