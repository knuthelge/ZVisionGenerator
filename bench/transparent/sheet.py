"""Lay mattes out for viewing: one row per subject, the render first, then one column per method."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_BACKDROPS = ((0, 0, 0), (255, 255, 255), (255, 0, 255), (40, 120, 40))
_LABEL_HEIGHT = 24


def backdrop(size: int) -> Image.Image:
    """Return a square split into black, white, magenta and green quadrants, so a fringe shows on at least one."""
    image = Image.new("RGB", (size, size))
    half = size // 2
    for index, colour in enumerate(_BACKDROPS):
        x, y = (index % 2) * half, (index // 2) * half
        image.paste(colour, (x, y, x + half, y + half))
    return image


def tile(path: Path, size: int, box: tuple[float, float, float, float] | None = None) -> Image.Image:
    """Return the image at *path* fitted into a *size* square, composited over :func:`backdrop` when it has alpha.

    *box* crops a region first, as fractions ``(left, top, right, bottom)``, for a close look at edges.
    """
    image = Image.open(path)
    if box is not None:
        image = image.crop(tuple(int(edge * dim) for edge, dim in zip(box, (image.width, image.height) * 2)))
    image = image.resize((size, size), Image.Resampling.LANCZOS) if box is not None else image
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    cell = backdrop(size) if image.mode == "RGBA" else Image.new("RGB", (size, size), (24, 24, 24))
    offset = ((size - image.width) // 2, (size - image.height) // 2)
    cell.paste(image, offset, image if image.mode == "RGBA" else None)
    return cell


def contact_sheet(rows: list[tuple[str, list[Path | None]]], columns: list[str], *, size: int = 256, box: tuple[float, float, float, float] | None = None) -> Image.Image:
    """Return a sheet with a header of *columns* and one row of tiles per ``(subject, paths)``; a missing path stays blank."""
    width, height = 100 + size * len(columns), _LABEL_HEIGHT + size * len(rows)
    sheet = Image.new("RGB", (width, height), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    for index, column in enumerate(columns):
        draw.text((100 + index * size + 6, 6), column, fill=(230, 230, 230))
    for row, (subject, paths) in enumerate(rows):
        top = _LABEL_HEIGHT + row * size
        draw.text((6, top + size // 2), subject, fill=(230, 230, 230))
        for index, path in enumerate(paths):
            if path is not None:
                sheet.paste(tile(path, size, box), (100 + index * size, top))
    return sheet


def to_rgba(alpha: np.ndarray, foreground: np.ndarray) -> Image.Image:
    """Return an RGBA image from float *alpha* and *foreground* in [0, 1]."""
    rgba = np.dstack([foreground, alpha[..., None]])
    return Image.fromarray((np.clip(rgba, 0.0, 1.0) * 255).round().astype(np.uint8), "RGBA")
