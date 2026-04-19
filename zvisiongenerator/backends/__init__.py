"""Backend registry — platform detection and backend lookup."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zvisiongenerator.core.image_backend import ImageBackend

__all__ = ["get_backend", "get_video_backend"]

BACKENDS: dict[str, "ImageBackend"] = {}

_PLATFORM_BACKEND_KEY: dict[str, str] = {
    "darwin": "mflux",
    "win32": "diffusers",
}


def _unsupported_platform_error(capability: str) -> RuntimeError:
    """Build RuntimeError with config-sourced platform labels."""
    from zvisiongenerator.utils.config import load_config
    from zvisiongenerator.utils.platform import get_platform_info, get_platform_label

    config = load_config()
    info = get_platform_info(config)
    supported_labels = [get_platform_label(config, k) for k in _PLATFORM_BACKEND_KEY]
    return RuntimeError(f"{capability.title()} generation is not supported on {info.label}. Supported: {', '.join(supported_labels)}.")


def _register_platform_backend() -> None:
    if sys.platform == "darwin":
        from zvisiongenerator.backends.image_mac import MfluxBackend

        BACKENDS["mflux"] = MfluxBackend()
    elif sys.platform == "win32":
        from zvisiongenerator.backends.image_win import DiffusersBackend

        BACKENDS["diffusers"] = DiffusersBackend()
    else:
        raise _unsupported_platform_error("image")


def get_backend() -> "ImageBackend":
    """Get the platform-appropriate backend."""
    if not BACKENDS:
        _register_platform_backend()
    key = _PLATFORM_BACKEND_KEY.get(sys.platform)
    if key is None:
        raise _unsupported_platform_error("image")
    return BACKENDS[key]


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
        raise _unsupported_platform_error("video")


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
