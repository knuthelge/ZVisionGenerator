"""Reference images: parse the CLI list, check it against what a model takes, and load the files."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from zvisiongenerator.utils.input_images import IMAGE_EXTENSIONS, open_image_upright
from zvisiongenerator.utils.paths import is_existing_file

# Both the diffusers and the mflux Klein pipelines scale a reference down to this many pixels, so a larger one only costs load time.
MAX_REFERENCE_PIXELS = 1024 * 1024


def parse_reference_paths(value: str) -> list[str]:
    """Split a comma-separated list of image paths, expanding ``~`` and dropping empty items."""
    return [str(Path(item.strip()).expanduser()) for item in value.split(",") if item.strip()]


def check_reference_count(family: str, count: int, max_images: int) -> None:
    """Raise ``ValueError`` when *count* references are more than a model family takes.

    Args:
        family: The model family, named in the message.
        count: How many references were given.
        max_images: How many the model takes; 0 means none.
    """
    if count == 0:
        return
    if max_images <= 0:
        raise ValueError(f"The {family} model does not support reference images.")
    if count > max_images:
        raise ValueError(f"The {family} model takes at most {max_images} reference images, got {count}.")


def check_reference_files(paths: Sequence[str]) -> None:
    """Raise ``ValueError`` for a reference that is not an existing PNG, JPEG or WebP file."""
    for path in paths:
        if not os.path.isfile(path):
            raise ValueError(f"Reference image not found: {path}")
        if Path(path).suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Reference image must be a PNG, JPEG or WebP file: {path}")


def missing_reference_files(paths: Sequence[str]) -> list[str]:
    """Return the *paths* that are not an existing file, in order and without repeats.

    A path the OS cannot look up counts as missing, and ``~`` is expanded as the CLI does.
    """
    missing: list[str] = []
    for path in paths:
        if not is_existing_file(path) and path not in missing:
            missing.append(path)
    return missing


def load_reference_image(path: str) -> Image.Image:
    """Return the reference at *path* as an upright RGB image of at most 1 MP.

    The photo's EXIF orientation is applied, transparency is composited over opaque white (as mflux does, so the
    pixels under a transparent area do not leak in), and an image larger than 1 MP is scaled down keeping its aspect.
    A reference is never cropped or resized to the output size; the model sizes it.

    Raises:
        FileNotFoundError: When *path* is not a file.
        ValueError: When the file is not a readable image.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Reference image not found: {path}")
    try:
        with Image.open(path) as probe:
            probe.verify()
        return _fit_area(open_image_upright(path))
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Reference is not a readable image: {path} ({exc})") from exc


def _fit_area(image: Image.Image) -> Image.Image:
    """Return *image* scaled down to at most :data:`MAX_REFERENCE_PIXELS`, keeping its aspect; smaller ones are unchanged."""
    width, height = image.size
    if width * height <= MAX_REFERENCE_PIXELS:
        return image
    scale = math.sqrt(MAX_REFERENCE_PIXELS / (width * height))
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
