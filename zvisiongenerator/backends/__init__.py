"""Backend registry — platform detection and backend lookup."""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession
    from zvisiongenerator.converters.ltx_checkpoint import LtxCheckpoint
    from zvisiongenerator.core.image_backend import ImageBackend
    from zvisiongenerator.core.prompt_enhancer import PromptEnhancer
    from zvisiongenerator.core.seedvr2 import SeedVR2Upscaler
    from zvisiongenerator.utils.model_memory import LtxCudaMemoryEstimate, MemoryBudget

__all__ = [
    "convert_ltx_checkpoint",
    "create_prompt_enhancer",
    "accelerator_memory_budget_or_none",
    "estimate_ltx_system_memory",
    "get_accelerator_memory_budget",
    "get_backend",
    "get_backend_name",
    "get_prompt_enhancer_session",
    "get_seedvr2_upscaler",
    "get_video_backend",
    "release_accelerator_memory",
    "supports_seedvr2",
    "supports_stored_quants",
    "supports_video_upscale_review",
]

BACKENDS: dict[str, "ImageBackend"] = {}


def _create_mflux_backend() -> "ImageBackend":
    from zvisiongenerator.backends.mac.image import MfluxBackend

    return MfluxBackend()


def _create_diffusers_backend() -> "ImageBackend":
    from zvisiongenerator.backends.cuda.image.backend import DiffusersBackend

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
    from zvisiongenerator.backends.mac.video import LtxVideoBackend

    return LtxVideoBackend()


def _create_diffusers_video_backend() -> "VideoBackend":
    from zvisiongenerator.backends.cuda.video.backend import DiffusersVideoBackend

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


# --- SeedVR2 upscaler ---


def _create_mflux_seedvr2_upscaler() -> "SeedVR2Upscaler":
    from zvisiongenerator.backends.mac.seedvr2 import MfluxSeedVR2Upscaler

    return MfluxSeedVR2Upscaler()


def _create_torch_seedvr2_upscaler() -> "SeedVR2Upscaler":
    from zvisiongenerator.backends.cuda.seedvr2 import TorchSeedVR2Upscaler

    return TorchSeedVR2Upscaler()


_SEEDVR2_UPSCALERS_MAP: dict[str, Callable[[], "SeedVR2Upscaler"]] = {
    "darwin": _create_mflux_seedvr2_upscaler,
    "win32": _create_torch_seedvr2_upscaler,
    "linux": _create_torch_seedvr2_upscaler,
}

_SEEDVR2_UPSCALERS: dict[str, "SeedVR2Upscaler"] = {}


def supports_seedvr2() -> bool:
    """Return whether this platform has a SeedVR2 upscaler (mflux on macOS, the vendored PyTorch core on Windows and Linux)."""
    return sys.platform in _SEEDVR2_UPSCALERS_MAP


def get_seedvr2_upscaler() -> "SeedVR2Upscaler | None":
    """Return the process-wide SeedVR2 upscaler, or ``None`` when this platform has none."""
    factory = _SEEDVR2_UPSCALERS_MAP.get(sys.platform)
    if factory is None:
        return None
    if sys.platform not in _SEEDVR2_UPSCALERS:
        _SEEDVR2_UPSCALERS[sys.platform] = factory()
    return _SEEDVR2_UPSCALERS[sys.platform]


# --- Accelerator memory ---


def release_accelerator_memory() -> None:
    """Return a finished job's model memory to the system (MLX's buffer cache on macOS; collected models on CUDA)."""
    if sys.platform in ("win32", "linux"):
        from zvisiongenerator.backends.cuda.memory import release_memory as release_cuda_memory

        release_cuda_memory()
        return
    if sys.platform != "darwin":
        return
    from zvisiongenerator.backends.mac.memory import release_memory

    release_memory()


def get_accelerator_memory_budget() -> MemoryBudget | None:
    """Return the memory a model may use on this machine, or ``None`` when unknown.

    Apple Silicon shares one memory pool between CPU and GPU, so its budget is Apple's recommended GPU
    working set. On CUDA it is the GPU's VRAM plus the system memory that offloaded weights stream from.
    """
    from zvisiongenerator.utils.model_memory import MemoryBudget

    if sys.platform == "darwin":
        from zvisiongenerator.backends.mac.memory import memory_budget_bytes, memory_size_bytes

        budget = memory_budget_bytes()
        return MemoryBudget(budget, unified_bytes=memory_size_bytes()) if budget else None
    if sys.platform in ("win32", "linux"):
        from zvisiongenerator.backends.cuda.memory import memory_sizes

        sizes = memory_sizes()
        return MemoryBudget(*sizes) if sizes else None
    return None


def supports_stored_quants() -> bool:
    """Return whether this platform's image backend saves quantized models for reuse (every supported platform)."""
    return sys.platform in _IMAGE_BACKENDS_MAP


@functools.cache
def accelerator_memory_budget_or_none() -> "MemoryBudget | None":
    """Return this machine's memory budget, read once per process (it is fixed hardware), or ``None`` when unknown or the probe fails."""
    try:
        return get_accelerator_memory_budget()
    except Exception:  # noqa: BLE001 - a missing or broken MLX or CUDA install means no budget, never a failed run
        return None


def supports_video_audio() -> bool:
    """Return whether this platform's video backend writes an audio track (every registered LTX backend does)."""
    return sys.platform in _VIDEO_BACKENDS_MAP


def supports_video_extend() -> bool:
    """Return whether this platform's video backend can extend a clip (every registered LTX backend can)."""
    return sys.platform in _VIDEO_BACKENDS_MAP


def supports_video_upscale_review() -> bool:
    """Return whether this platform's video backend can pause after the half-size render of an upscale (every registered LTX backend can)."""
    return sys.platform in _VIDEO_BACKENDS_MAP


def estimate_ltx_system_memory(config: dict[str, Any], resolved_model: str, quantize: int, *, find_local_dir: Callable[[str], Path | None]) -> "LtxCudaMemoryEstimate | None":
    """Return the CUDA video backend's system-memory estimate of LTX from files on disk, or ``None`` elsewhere or when files are missing."""
    if sys.platform not in ("win32", "linux"):
        return None
    from zvisiongenerator.backends.cuda.video.memory_estimate import cached_ltx_system_estimate

    return cached_ltx_system_estimate(config, resolved_model, quantize, find_local_dir=find_local_dir)


def convert_ltx_checkpoint(
    checkpoint: "LtxCheckpoint",
    target: "Path",
    bits: int,
    *,
    config: dict[str, Any],
    models_dir: "Path",
    lora: "Path | None",
    report: Callable[[str, int, int], None],
) -> None:
    """Write this platform's converted LTX-2.3 model folder of *checkpoint* at *bits* into the new folder *target*.

    Windows and Linux write a diffusers folder with an NF4 or FP8 transformer.

    Raises:
        RuntimeError: On macOS, which has no LTX converter yet, or when the conversion fails.
    """
    if sys.platform in ("win32", "linux"):
        from zvisiongenerator.backends.cuda.video.converted import convert

        convert(checkpoint, target, bits, config=config, models_dir=models_dir, lora=lora, report=report)
        return
    raise RuntimeError("Converting LTX checkpoints is only available on Windows and Linux so far.")


# --- Prompt enhancer ---


def create_prompt_enhancer(repo: str, revision: str | None) -> "PromptEnhancer":
    """Load the platform prompt-enhancer LLM (mlx-lm on macOS, transformers elsewhere)."""
    if sys.platform == "darwin":
        from zvisiongenerator.backends.mac.prompt_enhancer import MlxPromptEnhancer

        return MlxPromptEnhancer(repo, revision)
    if sys.platform in ("win32", "linux"):
        from zvisiongenerator.backends.cuda.prompt_enhancer import TransformersPromptEnhancer

        return TransformersPromptEnhancer(repo, revision)
    raise RuntimeError(f"Unsupported platform: {sys.platform}. Prompt enhancement supports macOS, Windows, and Linux.")


def get_prompt_enhancer_session() -> "PromptEnhancerSession":
    """Return the process-wide prompt-enhancer session."""
    from zvisiongenerator.backends.prompt_enhancer_session import get_prompt_enhancer_session as _get_session

    return _get_session()
