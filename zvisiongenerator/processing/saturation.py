"""Apply saturation adjustments to PIL images."""

from __future__ import annotations

from PIL import Image
from PIL.ImageEnhance import Color


def adjust_saturation(image: Image.Image, amount: float = 1.0) -> Image.Image:
    if amount < 0:
        raise ValueError(f"saturation amount must be >= 0, got {amount}")
    if amount == 1.0:
        return image
    return Color(image).enhance(amount)
