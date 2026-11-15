"""Rotate, flip and crop an upright PIL image the way the viewer's Adjust editor frames it."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

_ROTATIONS = (0, 90, 180, 270)
_TRANSPOSE_FOR_ROTATION = {
    90: Image.Transpose.ROTATE_270,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}


@dataclass(frozen=True)
class CropBox:
    """A crop rectangle in pixels of the rotated and flipped frame."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Reframe:
    """Clockwise quarter turns (in degrees), then a horizontal flip, then an optional crop."""

    rotate: int = 0
    flip: bool = False
    crop: CropBox | None = None

    def __post_init__(self) -> None:
        if self.rotate not in _ROTATIONS:
            raise ValueError(f"rotate must be one of {_ROTATIONS}, got {self.rotate}")


def framed_size(width: int, height: int, rotate: int) -> tuple[int, int]:
    """Return the (width, height) of an upright `width` x `height` picture after `rotate` degrees clockwise."""
    return (height, width) if rotate in (90, 270) else (width, height)


def reframe(image: Image.Image, frame: Reframe) -> Image.Image:
    """Return a new image rotated clockwise, flipped left to right, then cropped.

    Works in any mode, so alpha moves with the colour channels.

    Args:
        image: An upright picture (EXIF orientation already applied).
        frame: The rotation, flip and crop to apply.

    Returns:
        The framed image.

    Raises:
        ValueError: If the crop does not lie inside the rotated frame.
    """
    crop = frame.crop
    if crop is not None:
        _check_crop(crop, framed_size(image.width, image.height, frame.rotate))
    framed = image
    if frame.rotate:
        framed = framed.transpose(_TRANSPOSE_FOR_ROTATION[frame.rotate])
    if frame.flip:
        framed = framed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if crop is not None:
        framed = framed.crop((crop.x, crop.y, crop.x + crop.width, crop.y + crop.height))
    return framed.copy() if framed is image else framed


def _check_crop(crop: CropBox, size: tuple[int, int]) -> None:
    width, height = size
    inside = crop.x >= 0 and crop.y >= 0 and crop.width > 0 and crop.height > 0
    if not inside or crop.x + crop.width > width or crop.y + crop.height > height:
        raise ValueError("The crop is outside the picture.")
