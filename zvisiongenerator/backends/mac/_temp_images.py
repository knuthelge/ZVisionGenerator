"""Temporary PNG files for mflux, which reads its start and reference images from paths."""

from __future__ import annotations

import contextlib
import os
import tempfile

from PIL import Image

__all__ = ["remove_file", "save_temp_png"]


def save_temp_png(image: Image.Image) -> str:
    """Write *image* to a temporary PNG and return its path; the caller removes it with :func:`remove_file`."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.save(f, format="PNG")
        return f.name


def remove_file(path: str) -> None:
    """Delete a temporary file, ignoring one that is already gone."""
    with contextlib.suppress(OSError):
        os.unlink(path)
