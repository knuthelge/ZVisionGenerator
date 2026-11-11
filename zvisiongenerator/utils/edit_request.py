"""Parse and check the inputs of an image edit: the Extend request, the painted mask and the refusals the CLI and the Web UI share."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, UnidentifiedImageError

__all__ = [
    "EditMode",
    "check_edit_inputs",
    "decode_paint",
    "edit_mode",
    "image_is_cutout",
    "load_paint",
    "parse_extend",
    "source_is_cutout",
]

EXTEND_SIDES = ("top", "right", "bottom", "left")
# How far a picture may grow on one side, in pixels, and the step of the amount.
EXTEND_MIN = 128
EXTEND_MAX = 512
EXTEND_STEP = 16
# A pixel is transparent when its alpha is below this; a picture with any such pixel is a cut-out.
_OPAQUE_ALPHA = 250

EditMode = Literal["paint", "extend", "both", "whole"]


def parse_extend(value: str | Sequence[str | int]) -> tuple[str, int]:
    """Return the Extend request in *value* as a ``(side, amount)`` pair: *value* is ``"top=256"`` or such a pair.

    Raises:
        ValueError: The side is not top, right, bottom or left, or the amount is not a whole number from 128 to 512 in steps of 16.
    """
    if isinstance(value, str):
        side, separator, amount = value.partition("=")
        if not separator:
            raise ValueError(f"Extend takes a side and an amount, such as top=256, got {value!r}.")
    else:
        if len(value) != 2:
            raise ValueError(f"Extend takes a side and an amount, got {tuple(value)!r}.")
        side, amount = value
    side = str(side).strip().lower()
    if side not in EXTEND_SIDES:
        raise ValueError(f"Extend side must be top, right, bottom or left, got {side!r}.")
    if isinstance(amount, bool) or not isinstance(amount, (int, str)):
        raise ValueError(f"Extend amount must be a whole number of pixels, got {amount!r}.")
    try:
        pixels = int(amount)
    except ValueError as error:
        raise ValueError(f"Extend amount must be a whole number of pixels, got {amount!r}.") from error
    if not EXTEND_MIN <= pixels <= EXTEND_MAX or pixels % EXTEND_STEP:
        raise ValueError(f"Extend amount must be from {EXTEND_MIN} to {EXTEND_MAX} pixels in steps of {EXTEND_STEP}, got {pixels}.")
    return side, pixels


def _painted(image: Image.Image) -> np.ndarray:
    """Return the painted pixels of a decoded mask: alpha above 0 when it has a transparent pixel, else any channel above 0."""
    if image.has_transparency_data:
        alpha = np.asarray(image.convert("RGBA").getchannel("A"))
        if alpha.min() < 255:
            return alpha > 0
    if image.mode.startswith("I"):
        return np.asarray(image) > 0
    return np.asarray(image.convert("RGB")).any(axis=2)


def _scales_overlap(coarse: tuple[int, int], fine: tuple[int, int]) -> bool:
    """Return whether one scale, rounded to whole pixels on both axes, turns *fine* into *coarse*.

    Each axis allows the scales within half a pixel of *coarse* either way; the sizes match when those two ranges meet.
    """
    (coarse_width, coarse_height), (fine_width, fine_height) = coarse, fine
    low = max((coarse_width - 0.5) / fine_width, (coarse_height - 0.5) / fine_height)
    high = min((coarse_width + 0.5) / fine_width, (coarse_height + 0.5) / fine_height)
    return low <= high + 1e-9


def _aspect_matches(mask_size: tuple[int, int], size: tuple[int, int]) -> bool:
    """Return whether a mask of *mask_size* is *size* scaled by one factor, give or take the rounding of the coarser grid on each axis."""
    if mask_size[0] >= size[0]:
        return _scales_overlap(size, mask_size)
    return _scales_overlap(mask_size, size)


def load_paint(mask_path: str, width: int, height: int) -> np.ndarray:
    """Return the painted area of the PNG file at *mask_path* as a bool array of *height* by *width*.

    Raises:
        FileNotFoundError: *mask_path* does not exist.
        ValueError: The file is not a readable PNG, its shape does not match, or nothing is painted (see :func:`decode_paint`).
    """
    path = Path(mask_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"The mask {mask_path} does not exist.")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"The mask is not a readable PNG: {error}") from error
    return decode_paint(data, width, height)


def decode_paint(data: bytes, width: int, height: int) -> np.ndarray:
    """Return the painted area of the PNG in *data* as a bool array of *height* by *width*.

    When the PNG has a transparent pixel, a pixel is painted when its alpha is above 0; otherwise, an opaque PNG with or
    without an alpha channel, when any colour channel is above 0 (not black). The mask may be another size of the same
    shape (the browser sends at most 4 MP), give or take the rounding of each side, and is scaled with nearest neighbour.

    Raises:
        ValueError: *data* is not a readable PNG, its shape does not match, or nothing is painted.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG":
                raise ValueError("The mask must be a PNG.")
            painted = _painted(image)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"The mask is not a readable PNG: {error}") from error
    mask_height, mask_width = painted.shape
    if (mask_width, mask_height) != (width, height):
        if not _aspect_matches((mask_width, mask_height), (width, height)):
            raise ValueError(f"The mask is {mask_width}x{mask_height}, which does not match the shape of the {width}x{height} image.")
        scaled = Image.fromarray(painted.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
        painted = np.asarray(scaled) > 0
    if not painted.any():
        raise ValueError("The mask has no painted area.")
    return painted


def edit_mode(has_mask: bool, extend: tuple[str, int] | None) -> EditMode:
    """Return what an edit does: ``paint``, ``extend``, ``both`` (paint on the extended picture) or ``whole`` (no paint, no Extend)."""
    if has_mask:
        return "both" if extend else "paint"
    return "extend" if extend else "whole"


def image_is_cutout(image: Image.Image) -> bool:
    """Return whether *image* has a transparent background: any pixel with alpha below 250."""
    if not image.has_transparency_data:
        return False
    low, _ = image.convert("RGBA").getchannel("A").getextrema()
    return low < _OPAQUE_ALPHA


def source_is_cutout(path: str) -> bool:
    """Return whether the image at *path* has a transparent background (see :func:`image_is_cutout`)."""
    with Image.open(path) as image:
        return image_is_cutout(image)


def check_edit_inputs(mode: EditMode, extend: tuple[str, int] | None, cutout: bool, max_references: int, supports_mask_edit: bool = True) -> None:
    """Raise ``ValueError`` when an edit cannot run, in words the CLI and the Web UI both show.

    Args:
        mode: What the edit does, from :func:`edit_mode`.
        extend: The Extend request, if any.
        cutout: Whether the source has a transparent background.
        max_references: How many references the model takes; 0 means none.
        supports_mask_edit: Whether the model can redraw a painted area.
    """
    if not supports_mask_edit:
        raise ValueError("This model cannot edit a picture. Pick a model that can.")
    if extend is not None and cutout:
        raise ValueError("A picture with a transparent background cannot be extended.")
    if mode == "whole" and (cutout or max_references <= 0):
        raise ValueError("Paint the area to change.")
