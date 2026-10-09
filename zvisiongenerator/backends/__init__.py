"""Backend registry — platform detection and backend lookup."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession
    from zvisiongenerator.core.image_backend import ImageBackend
    from zvisiongenerator.core.prompt_enhancer import PromptEnhancer

__all__ = [
    "create_prompt_enhancer",
    "get_accelerator_memory_budget",
    "get_backend",
    "get_backend_name",
    "get_prompt_enhancer_session",
    "get_video_backend",
    "release_accelerator_memory",
    "supports_stored_quants",
]

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
    "linux": ("diffusers", _create_diffusers_backend),
}


def _get_image_backend_registration() -> tuple[str, Callable[[], "ImageBackend"]]:
    try:
        return _IMAGE_BACKENDS_MAP[sys.platform]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported platform: {sys.platform}. Image generation supports macOS, Windows, and Linux.") from exc


def _register_platform_backend() -> None:
    backend_key, factory = _get_image_backend_registration()
    BACKENDS[backend_key] = factory()


def get_backend() -> "ImageBackend":
    """Get the platform-appropriate backend."""
    if not BACKENDS:
        _register_platform_backend()
    backend_key, _ = _get_image_backend_registration()
    return BACKENDS[backend_key]


def get_backend_name() -> str:
    """Return the registered image backend key for the current platform."""
    backend_key, _ = _get_image_backend_registration()
    return backend_key


# --- Video backends ---

if TYPE_CHECKING:
    from zvisiongenerator.core.video_backend import VideoBackend

VIDEO_BACKENDS: dict[str, "VideoBackend"] = {}


def _create_ltx_video_backend() -> "VideoBackend":
    from zvisiongenerator.backends.video_mac import LtxVideoBackend

    return LtxVideoBackend()


def _create_diffusers_video_backend() -> "VideoBackend":
    from zvisiongenerator.backends.video_diffusers import DiffusersVideoBackend

    return DiffusersVideoBackend()


_VIDEO_BACKENDS_MAP: dict[str, tuple[str, Callable[[], "VideoBackend"]]] = {
    "darwin": ("ltx", _create_ltx_video_backend),
    "win32": ("ltx", _create_diffusers_video_backend),
    "linux": ("ltx", _create_diffusers_video_backend),
}


def _get_video_backend_registration() -> tuple[str, Callable[[], "VideoBackend"]]:
    try:
        return _VIDEO_BACKENDS_MAP[sys.platform]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported video platform: {sys.platform}. Video generation supports macOS, Windows, and Linux.") from exc


def _register_video_backends() -> None:
    """Register the platform video backend for the active OS."""
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


# --- Accelerator memory ---


def release_accelerator_memory() -> None:
    """Return a finished job's model memory to the system (MLX's buffer cache on macOS; collected models on CUDA)."""
    if sys.platform in ("win32", "linux"):
        from zvisiongenerator.backends.memory_cuda import release_memory as release_cuda_memory

        release_cuda_memory()
        return
    if sys.platform != "darwin":
        return
    from zvisiongenerator.backends.memory_mac import release_memory

    release_memory()


def get_accelerator_memory_budget() -> int | None:
    """Return the GPU memory a model may use without starving the system, or ``None`` when unknown.

    Only macOS reports this: Apple Silicon shares one memory pool between CPU and GPU, so the
    budget is Apple's recommended working set. CUDA has dedicated VRAM and is not estimated here.
    """
    if sys.platform != "darwin":
        return None
    from zvisiongenerator.backends.memory_mac import memory_budget_bytes

    return memory_budget_bytes()


def supports_stored_quants() -> bool:
    """Return whether this platform's image backend saves quantized models for reuse (macOS/mflux only)."""
    return sys.platform == "darwin"


# --- Prompt enhancer ---


def create_prompt_enhancer(repo: str, revision: str | None) -> "PromptEnhancer":
    """Load the platform prompt-enhancer LLM (mlx-lm on macOS, transformers elsewhere)."""
    if sys.platform == "darwin":
        from zvisiongenerator.backends.prompt_enhancer_mac import MlxPromptEnhancer

        return MlxPromptEnhancer(repo, revision)
    if sys.platform in ("win32", "linux"):
        from zvisiongenerator.backends.prompt_enhancer_win import TransformersPromptEnhancer

        return TransformersPromptEnhancer(repo, revision)
    raise RuntimeError(f"Unsupported platform: {sys.platform}. Prompt enhancement supports macOS, Windows, and Linux.")


def get_prompt_enhancer_session() -> "PromptEnhancerSession":
    """Return the process-wide prompt-enhancer session."""
    from zvisiongenerator.backends.prompt_enhancer_session import get_prompt_enhancer_session as _get_session

    return _get_session()
