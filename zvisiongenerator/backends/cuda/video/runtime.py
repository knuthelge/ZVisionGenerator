"""Helpers shared by the CUDA video backends: torch runtime tuning and fitting a conditioning image to a canvas."""

from __future__ import annotations

from typing import Any

__all__ = ["configure_torch_runtime", "fit_image"]


def configure_torch_runtime(torch: Any) -> None:
    """Apply the same lightweight CUDA tuning used by the image diffusers backend."""
    from zvisiongenerator.backends.cuda.memory import configure_allocator

    configure_allocator()
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def fit_image(image: Any, width: int, height: int) -> Any:
    """Scale *image* to cover *width* × *height* with its aspect ratio kept, and crop the overflow from the center.

    The diffusers pipeline would otherwise stretch the image to the video's size; the macOS pipeline crops the same way.
    """
    from PIL import Image, ImageOps

    return ImageOps.fit(image, (width, height), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))
