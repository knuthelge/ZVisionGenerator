"""Shared helpers that write small input images a phone or an editor would produce: sideways photos and cut-outs."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# Stored 40 x 20 with a red left half and a blue right half, tagged "rotate 90 degrees clockwise" to show upright:
# 20 wide and 40 high, red on top and blue below.
SIDEWAYS_STORED_SIZE = (40, 20)
SIDEWAYS_UPRIGHT_SIZE = (20, 40)


def write_sideways_photo(path: Path) -> str:
    """Write a JPEG whose EXIF orientation tag turns it upright, and return its path."""
    photo = Image.new("RGB", SIDEWAYS_STORED_SIZE, (255, 0, 0))
    photo.paste((0, 0, 255), (20, 0, 40, 20))
    exif = Image.Exif()
    exif[0x0112] = 6
    photo.save(path, quality=100, subsampling=0, exif=exif)
    return str(path)


def write_cutout(path: Path) -> str:
    """Write a PNG with an opaque red left half and a fully transparent right half (blue underneath), and return its path."""
    cutout = Image.new("RGBA", (8, 4), (255, 0, 0, 255))
    cutout.paste((0, 0, 255, 0), (4, 0, 8, 4))
    cutout.save(path)
    return str(path)


def is_reddish(pixel: tuple[int, ...]) -> bool:
    """Return whether *pixel* is red, allowing for JPEG loss."""
    return pixel[0] > 200 and pixel[1] < 60 and pixel[2] < 60


def is_bluish(pixel: tuple[int, ...]) -> bool:
    """Return whether *pixel* is blue, allowing for JPEG loss."""
    return pixel[2] > 200 and pixel[0] < 60 and pixel[1] < 60
