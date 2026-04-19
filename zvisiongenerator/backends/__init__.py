"""Backend registry — platform detection and backend lookup."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zvisiongenerator.core.image_backend import ImageBackend

__all__ = ["get_backend", "get_video_backend"]

BACKENDS: dict[str, "ImageBackend"] = {}


def _register_platform_backend() -> None:
    if sys.platform == "darwin":
        from zvisiongenerator.backends.image_mac import MfluxBackend

        BACKENDS["mflux"] = MfluxBackend()
    elif sys.platform == "win32":
        from zvisiongenerator.backends.image_win import DiffusersBackend

        BACKENDS["diffusers"] = DiffusersBackend()
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}. ZVisionGenerator supports macOS and Windows.")


def get_backend() -> "ImageBackend":
    """Get the platform-appropriate backend."""
    if not BACKENDS:
        _register_platform_backend()
    if sys.platform == "darwin":
        return BACKENDS["mflux"]
    elif sys.platform == "win32":
        return BACKENDS["diffusers"]
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}. ZVisionGenerator supports macOS and Windows.")


# --- Video backends ---

if TYPE_CHECKING:
    from zvisiongenerator.core.video_backend import VideoBackend

VIDEO_BACKENDS: dict[str, "VideoBackend"] = {}


def _register_video_backends() -> None:
    """Register available video backends."""
    if sys.platform == "darwin":
        from zvisiongenerator.backends.video_mac import LtxVideoBackend

        VIDEO_BACKENDS["ltx"] = LtxVideoBackend()
    elif sys.platform == "win32":
        from zvisiongenerator.backends.video_win import LtxCudaVideoBackend

        VIDEO_BACKENDS["ltx"] = LtxCudaVideoBackend()
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}. ZVisionGenerator supports macOS and Windows.")


def get_video_backend(family: str) -> "VideoBackend":
    """Get video backend by model family name.

    Args:
        family: Model family ("ltx").

    Returns:
        The video backend instance for that family.

    Raises:
        RuntimeError: If family is unknown or platform unsupported.
    """
    if not VIDEO_BACKENDS:
        _register_video_backends()
    if family not in VIDEO_BACKENDS:
        raise RuntimeError(f"No video backend for model family '{family}'. Available: {list(VIDEO_BACKENDS)}")
    return VIDEO_BACKENDS[family]
