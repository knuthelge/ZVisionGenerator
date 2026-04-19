"""Backend registry — platform detection and backend lookup."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zvisiongenerator.core.image_backend import ImageBackend

__all__ = ["get_backend", "get_video_backend"]

BACKENDS: dict[str, "ImageBackend"] = {}


def _create_mflux_backend() -> "ImageBackend":
    from zvisiongenerator.backends.image_mac import MfluxBackend

    return MfluxBackend()


def _create_diffusers_backend() -> "ImageBackend":
    from zvisiongenerator.backends.image_win import DiffusersBackend

    return DiffusersBackend()


_IMAGE_BACKENDS_MAP: dict[str, tuple[str, Callable[[], "ImageBackend"]]] = {
    "darwin": ("mflux", _create_mflux_backend),
    "win32": ("diffusers", _create_diffusers_backend),
}


def _get_image_backend_registration() -> tuple[str, Callable[[], "ImageBackend"]]:
    try:
        return _IMAGE_BACKENDS_MAP[sys.platform]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported platform: {sys.platform}. ZVisionGenerator supports macOS and Windows.") from exc


def _register_platform_backend() -> None:
    backend_key, factory = _get_image_backend_registration()
    BACKENDS[backend_key] = factory()


def get_backend() -> "ImageBackend":
    """Get the platform-appropriate backend."""
    if not BACKENDS:
        _register_platform_backend()
    backend_key, _ = _get_image_backend_registration()
    return BACKENDS[backend_key]


# --- Video backends ---

if TYPE_CHECKING:
    from zvisiongenerator.core.video_backend import VideoBackend

VIDEO_BACKENDS: dict[str, "VideoBackend"] = {}


def _create_ltx_video_backend() -> "VideoBackend":
    from zvisiongenerator.backends.video_mac import LtxVideoBackend

    return LtxVideoBackend()


_VIDEO_BACKENDS_MAP: dict[str, tuple[str, Callable[[], "VideoBackend"]]] = {
    "darwin": ("ltx", _create_ltx_video_backend),
}


def _get_video_backend_registration() -> tuple[str, Callable[[], "VideoBackend"]]:
    try:
        return _VIDEO_BACKENDS_MAP[sys.platform]
    except KeyError as exc:
        raise RuntimeError(f"Video generation is currently macOS-only. Current platform: {sys.platform}") from exc


def _register_video_backends() -> None:
    """Register available video backends. macOS only for now."""
    backend_key, factory = _get_video_backend_registration()
    VIDEO_BACKENDS[backend_key] = factory()


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
