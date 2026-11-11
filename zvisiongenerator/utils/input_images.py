"""Open an input image the way every consumer wants it: upright, on white and RGB.

A start image, a video's first or last frame and a reference all go through here.
"""

from __future__ import annotations

import io
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageCms, ImageOps

# The image types the app takes as an input: references, a start or last-frame image and an upscale source.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
_EXIF_ORIENTATION = 0x0112
# The orientation values that turn a picture a quarter turn, so its width and height swap.
_EXIF_QUARTER_TURNS = frozenset({5, 6, 7, 8})


def open_image_upright(path: str | Path) -> Image.Image:
    """Return the image at *path* as an upright RGB image, with any transparency composited over opaque white.

    The photo's EXIF orientation is applied, so a phone photo is not sideways. The size is kept: the caller fits it.
    """
    with Image.open(path) as opened:
        return flatten_alpha(ImageOps.exif_transpose(opened))


# The modes that are converted to RGB before their colour profile is applied; any other mode (CMYK) is managed as it is.
_RGB_LIKE_MODES = frozenset({"1", "L", "LA", "La", "P", "PA", "RGB", "RGBA", "RGBa"})


def to_srgb(image: Image.Image) -> Image.Image:
    """Return *image* in sRGB: converted from its embedded colour profile, with the profile dropped, and any alpha kept.

    An image without a profile is returned as it is. One whose profile cannot be used (a damaged or grey one) is
    returned unconverted, without the profile. CMYK is converted through its own profile, not through naive RGB.
    """
    profile = image.info.get("icc_profile")
    if not profile:
        return image
    info = {key: value for key, value in image.info.items() if key not in ("icc_profile", "transparency")}
    if image.mode in _RGB_LIKE_MODES:
        mode = "RGBA" if image.has_transparency_data else "RGB"
        source, output = image.convert(mode), mode
    else:
        source, output = image, "RGB"
    try:
        converted = ImageCms.profileToProfile(
            source,
            ImageCms.ImageCmsProfile(io.BytesIO(profile)),
            ImageCms.createProfile("sRGB"),
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            outputMode=output,
        )
    except ImageCms.PyCMSError, OSError:
        converted = image.copy()
    converted.info = info
    return converted


def upright_size(path: str | Path) -> tuple[int, int]:
    """Return the upright ``(width, height)`` of the image at *path*, reading only its header and EXIF orientation.

    Raises:
        OSError: The file cannot be read as an image.
    """
    with Image.open(path) as image:
        return image_upright_size(image)


def image_upright_size(image: Image.Image) -> tuple[int, int]:
    """Return the upright ``(width, height)`` of the opened *image*: its size, swapped when its EXIF orientation turns it a quarter."""
    width, height = image.size
    return (height, width) if image.getexif().get(_EXIF_ORIENTATION) in _EXIF_QUARTER_TURNS else (width, height)


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


def flatten_alpha(image: Image.Image, background: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Return *image* as RGB, with any transparency composited over *background* (white by default)."""
    if image.has_transparency_data:
        # Pillow cannot convert premultiplied "La" to RGBA directly.
        rgba = (image.convert("LA") if image.mode == "La" else image).convert("RGBA")
        backdrop = Image.new("RGBA", rgba.size, (*background, 255))
        return Image.alpha_composite(backdrop, rgba).convert("RGB")
    return image.convert("RGB")
