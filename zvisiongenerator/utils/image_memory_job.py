"""Build the image :class:`~zvisiongenerator.utils.memory_verdict.MemoryJob` an image load is told about."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from PIL import Image

from zvisiongenerator.processing.edit_region import edit_model_size
from zvisiongenerator.utils.lora_size import lora_files_bytes
from zvisiongenerator.utils.memory_verdict import MemoryJob, capped_reference_megapixels

__all__ = ["file_megapixels", "image_memory_job", "run_memory_job", "sized_edit_job", "sized_image_job", "starts_from_picture"]


def sized_image_job(width: int, height: int, *, reference_megapixels: Iterable[float] = (), lora_bytes: int = 0, from_picture: bool = False) -> MemoryJob:
    """Return the memory job of an image that draws *width* x *height*, holds references of *reference_megapixels* (one entry each) and applies LoRA files of *lora_bytes* in all.

    *from_picture* is set when the model starts from a picture at that size (an upscale's refine or an img2img). This is the one place the job of a plain
    image is assembled: the command line and the loader measure files and the Web UI is told sizes, and all of them come here.
    """
    return MemoryJob(media="image", width=width, height=height, reference_megapixels=tuple(reference_megapixels), lora_bytes=lora_bytes, from_picture=from_picture)


def sized_edit_job(source_size: tuple[int, int], *, masked: bool, extend: tuple[str, int] | None, lora_bytes: int = 0) -> MemoryJob:
    """Return the memory job of an edit of a picture of *source_size*: the canvas the model draws, with its one reference.

    The reference is the picture, except for an Extend, whose reference is the drawn canvas (the source plus the new strip), capped as the model caps a reference.
    A painted edit and an Extend redraw part of a picture at the canvas size, so they start from a picture; a whole-picture edit does not.
    """
    width, height = edit_model_size(*source_size, masked=masked, extend=extend)
    if extend is not None:
        reference_megapixels = capped_reference_megapixels((width * height / 1e6,))
    else:
        reference_megapixels = source_size[0] * source_size[1] / 1e6
    return sized_image_job(width, height, reference_megapixels=(reference_megapixels,), lora_bytes=lora_bytes, from_picture=masked or extend is not None)


def image_memory_job(width: int, height: int, *, reference_paths: Iterable[str | Path] = (), lora_paths: Sequence[str] | None = None, from_picture: bool = False) -> MemoryJob:
    """Return the memory job of an image that draws *width* x *height*, holds the pictures at *reference_paths* and applies *lora_paths*.

    A reference file that cannot be read counts nothing here: the job refuses it where it checks its inputs. See :func:`sized_image_job` for *from_picture*.
    """
    return sized_image_job(width, height, reference_megapixels=file_megapixels(reference_paths), lora_bytes=_lora_bytes(lora_paths), from_picture=from_picture)


def starts_from_picture(*, upscale: int | None, start_image: bool) -> bool:
    """Return whether a plain image job starts from a picture at the size it draws: an upscale refines its render, and a start image is an img2img."""
    return bool(upscale) or start_image


def run_memory_job(args: argparse.Namespace, *, size: Callable[[], tuple[int, int]], reference_paths: Iterable[str | Path], lora_paths: Sequence[str] | None) -> MemoryJob:
    """Return the memory job of an image run: an edit's canvas and picture when *args* carry an edit, else a plain image.

    A plain image starts from a picture when it has a start image (``image_path``) or an upscale, which refines the render at the final size.

    Args:
        args: The run's arguments; an edit sets ``edit_source``, ``edit_size``, ``edit_paint`` and ``edit_extend``.
        size: Returns the ``(width, height)`` a plain image draws; not called for an edit, which has no ratio or size.
        reference_paths: The reference pictures of a plain image.
        lora_paths: The LoRA files the run applies.
    """
    if getattr(args, "edit_source", None) is not None:
        return sized_edit_job(args.edit_size, masked=args.edit_paint is not None, extend=args.edit_extend, lora_bytes=_lora_bytes(lora_paths))
    width, height = size()
    from_picture = starts_from_picture(upscale=getattr(args, "upscale", None), start_image=getattr(args, "image_path", None) is not None)
    return image_memory_job(width, height, reference_paths=reference_paths, lora_paths=lora_paths, from_picture=from_picture)


def file_megapixels(paths: Iterable[str | Path]) -> tuple[float, ...]:
    """Return the decimal megapixels of each readable picture in *paths*, reading only its header."""
    sizes: list[float] = []
    for path in paths:
        try:
            with Image.open(path) as picture:
                sizes.append(picture.width * picture.height / 1e6)
        except OSError:
            continue
    return tuple(sizes)


def _lora_bytes(paths: Sequence[str] | None) -> int:
    """Return the total size of the LoRA files at *paths*; sizing the job never fails it, so a file that cannot be read counts nothing."""
    try:
        return lora_files_bytes(paths or ())
    except OSError:
        return 0
