"""Shrink a Hugging Face preview image to the small picture Discover keeps for a downloaded LoRA."""

from __future__ import annotations

import io

__all__ = ["MAX_PREVIEW_PIXELS", "PREVIEW_EDGE", "make_preview"]

# The longest edge of the stored picture, in pixels: sharp on a tile of the Installed tab, small enough to keep many.
PREVIEW_EDGE = 384
# The most pixels a source image may have; a larger one is refused before it is decoded.
MAX_PREVIEW_PIXELS = 40_000_000
_QUALITY = 82


def make_preview(data: bytes) -> tuple[str, bytes] | None:
    """Return ``("image/webp", bytes)`` of *data* shrunk to ``PREVIEW_EDGE`` on its longest side, or None when it is not a usable image.

    Args:
        data: The bytes of a PNG, JPEG, WebP or GIF image (the first frame of an animation).
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_PREVIEW_PIXELS:
                return None
            image.seek(0)
            small = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            small.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE))
            out = io.BytesIO()
            small.save(out, format="WEBP", quality=_QUALITY)
    except OSError, ValueError, SyntaxError, Image.DecompressionBombError:
        return None
    return "image/webp", out.getvalue()
