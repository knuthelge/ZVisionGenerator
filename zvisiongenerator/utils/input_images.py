"""Open an input image the way every consumer wants it: upright, on white and RGB.

A start image, a video's first or last frame and a reference all go through here.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageOps

# The image types the app takes as an input: references, a start or last-frame image and an upscale source.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
_EXIF_ORIENTATION = 0x0112


def open_image_upright(path: str | Path) -> Image.Image:
    """Return the image at *path* as an upright RGB image, with any transparency composited over opaque white.

    The photo's EXIF orientation is applied, so a phone photo is not sideways. The size is kept: the caller fits it.
    """
    with Image.open(path) as opened:
        return _flatten_alpha(ImageOps.exif_transpose(opened))


@contextmanager
def upright_image_copies(*paths: str | None) -> Iterator[list[str | None]]:
    """Yield *paths*, with each sideways or transparent image replaced by an upright opaque PNG copy.

    For a consumer that takes a file path and cannot be handed an image. An image that needs no change keeps its
    own path, and ``None`` stays ``None``. The copies are removed when the block ends.
    """
    with tempfile.TemporaryDirectory(prefix="ziv-upright-") as directory:
        resolved: list[str | None] = []
        for index, path in enumerate(paths):
            if path is None or not _needs_upright(path):
                resolved.append(path)
                continue
            copy = Path(directory) / f"{index}.png"
            open_image_upright(path).save(copy, format="PNG")
            resolved.append(str(copy))
        yield resolved


def _needs_upright(path: str) -> bool:
    """Return whether the image at *path* is stored sideways or has transparency."""
    with Image.open(path) as image:
        return image.getexif().get(_EXIF_ORIENTATION, 1) != 1 or image.has_transparency_data


def _flatten_alpha(image: Image.Image) -> Image.Image:
    """Return *image* as RGB, with any transparency composited over white."""
    if image.has_transparency_data:
        # Pillow cannot convert premultiplied "La" to RGBA directly.
        rgba = (image.convert("LA") if image.mode == "La" else image).convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")
