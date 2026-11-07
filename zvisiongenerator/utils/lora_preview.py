"""Shrink an image to the small WebP picture the Web UI keeps: Discover's LoRA previews and the file browser's thumbnails."""

from __future__ import annotations

import io
from pathlib import Path

__all__ = ["LARGE_PREVIEW_EDGE", "PREVIEW_EDGE", "make_preview"]

# The longest edge of the stored picture, in pixels: sharp on a tile of the Installed tab, small enough to keep many.
PREVIEW_EDGE = 384
# The longest edge of the larger picture: a gallery card, the strip and the file browser's preview pane.
LARGE_PREVIEW_EDGE = 768
_QUALITY = 82


def make_preview(source: bytes | Path, edge: int = PREVIEW_EDGE) -> tuple[str, bytes] | None:
    """Return ``("image/webp", bytes)`` of *source* shrunk to *edge* (``PREVIEW_EDGE`` by default) on its longest side, or None when it is not a usable image.

    The picture is turned upright by its EXIF orientation and carries no metadata (EXIF, GPS or the text a PNG
    embeds); transparency is kept. A path is opened without reading the file first. A JPEG is decoded at reduced
    size, and the pixel count that must stay under Pillow's ``Image.MAX_IMAGE_PIXELS`` is measured after that
    reduction; any other format is measured from its header. Pillow itself still refuses, when the file is opened,
    an image of more than twice that limit.

    Args:
        source: The bytes of a PNG, JPEG, WebP or GIF image (the first frame of an animation), or the path of one.
        edge: The longest side of the result, in pixels.
    """
    from PIL import Image, ImageOps

    try:
        with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as image:
            if image.format == "JPEG":
                image.draft("RGB", (edge, edge))
            if Image.MAX_IMAGE_PIXELS is not None and image.width * image.height > Image.MAX_IMAGE_PIXELS:
                return None
            image.seek(0)
            ImageOps.exif_transpose(image, in_place=True)
            has_alpha = "A" in image.getbands() or (image.mode == "P" and "transparency" in image.info)
            small = image.convert("RGBA" if has_alpha else "RGB")
            small.thumbnail((edge, edge))
            out = io.BytesIO()
            small.save(out, format="WEBP", quality=_QUALITY)
    except OSError, ValueError, SyntaxError, Image.DecompressionBombError:
        return None
    return "image/webp", out.getvalue()
