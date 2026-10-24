"""Backend registry — platform detection and backend lookup."""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any, cast

from zvisiongenerator.core.video_family import VideoCapabilities

if TYPE_CHECKING:
    from pathlib import Path

    from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession
    from zvisiongenerator.converters.ltx_checkpoint import LtxCheckpoint
    from zvisiongenerator.core.image_backend import ImageBackend
    from zvisiongenerator.core.prompt_enhancer import PromptEnhancer
    from zvisiongenerator.core.seedvr2 import SeedVR2Upscaler
    from zvisiongenerator.core.video_family import VideoFamily
    from zvisiongenerator.downloads.target import DownloadTarget
    from zvisiongenerator.utils.h3_preset import H3MlxPreset, H3Preset, HubFiles
    from zvisiongenerator.utils.model_memory import LtxCudaMemoryEstimate, MemoryBudget

__all__ = [
    "convert_ltx_checkpoint",
    "create_prompt_enhancer",
    "accelerator_memory_budget_or_none",
    "enhancer_download_plan",
    "estimate_ltx_system_memory",
    "get_accelerator_memory_budget",
    "get_backend",
    "get_backend_name",
    "get_prompt_enhancer_session",
    "get_seedvr2_upscaler",
    "get_video_backend",
    "h3_companion_repo_ids",
    "h3_download_plan",
    "h3_load_options",
    "image_download_plan",
    "release_accelerator_memory",
    "supports_seedvr2",
    "supports_stored_quants",
    "supports_video_upscale_review",
    "video_capabilities",
    "video_download_plan",
    "video_family_available",
    "video_family_on_platform",
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


def _create_h3_video_backend() -> "VideoBackend":
    from zvisiongenerator.backends.cuda.video.h3.backend import H3VideoBackend

    return H3VideoBackend()


def _create_h3_mlx_video_backend() -> "VideoBackend":
    from zvisiongenerator.backends.mac.h3_video import H3MlxVideoBackend

    return H3MlxVideoBackend()


# Per platform, the video backend factory of each model family.
_VIDEO_BACKENDS_MAP: dict[str, dict[str, Callable[[], "VideoBackend"]]] = {
    "darwin": {"ltx": _create_ltx_video_backend, "h3": _create_h3_mlx_video_backend},
    "win32": {"ltx": _create_diffusers_video_backend, "h3": _create_h3_video_backend},
    "linux": {"ltx": _create_diffusers_video_backend, "h3": _create_h3_video_backend},
}


def _get_video_backend_factories() -> dict[str, Callable[[], "VideoBackend"]]:
    try:
        return _VIDEO_BACKENDS_MAP[sys.platform]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported video platform: {sys.platform}. Video generation supports macOS, Windows, and Linux.") from exc


def get_video_backend(family: str) -> "VideoBackend":
    """Get the video backend of a model family on this platform, created on first use.

    Args:
        family: Model family ("ltx" or "h3").

    Returns:
        The video backend instance for that family.

    Raises:
        RuntimeError: If the platform is unsupported or has no backend for the family.
    """
    factories = _get_video_backend_factories()
    if family in VIDEO_BACKENDS:
        return VIDEO_BACKENDS[family]
    if family not in factories:
        raise RuntimeError(f"No video backend for model family '{family}'. Available: {list(factories)}")
    VIDEO_BACKENDS[family] = factories[family]()
    return VIDEO_BACKENDS[family]


def video_family_available(family: str) -> bool:
    """Return whether this platform has a video backend for the model family."""
    return family in _VIDEO_BACKENDS_MAP.get(sys.platform, {})


def video_capabilities(config: dict[str, Any], family: str) -> VideoCapabilities | None:
    """Return what the family can do on this platform, or ``None`` when the config does not describe the family.

    The family's capabilities, overlaid by its ``platform_capabilities`` for this platform, then limited by what this
    platform's video backend supports (audio, extend and upscale review) and by whether it has a backend for the
    family at all (without one nothing is possible).

    Raises:
        ValueError: If the family's description is malformed.
    """
    from zvisiongenerator.utils.config import video_family

    described = video_family(config, family)
    if described is None:
        return None
    flags = {**asdict(described.capabilities), **described.platform_capabilities.get(sys.platform, {})}
    available = video_family_available(family)
    limits = {"audio": supports_video_audio(), "extend": supports_video_extend(), "upscale_review": supports_video_upscale_review()}
    return VideoCapabilities(**{name: available and flag and limits.get(name, True) for name, flag in flags.items()})


def video_family_on_platform(config: dict[str, Any], family: str) -> "VideoFamily | None":
    """Return the description of a video family as this platform runs it, or ``None`` when the config does not describe it.

    The family's ``carry_frames`` are replaced by its ``platform_carry_frames`` entry for this platform, so the window an
    extension generates (and the estimate, rewrite context and request built from it) matches what the backend renders.

    Raises:
        ValueError: If the family's description is malformed.
    """
    from zvisiongenerator.utils.config import video_family

    described = video_family(config, family)
    if described is None or sys.platform not in described.platform_carry_frames:
        return described
    return replace(described, carry_frames=described.platform_carry_frames[sys.platform])


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


def estimate_ltx_system_memory(
    config: dict[str, Any],
    resolved_model: str,
    quantize: int,
    *,
    find_local_dir: Callable[[str], Path | None],
    model_name: str | None = None,
) -> "LtxCudaMemoryEstimate | None":
    """Return the CUDA video backend's system-memory estimate of LTX from files on disk, or ``None`` elsewhere or when files are missing.

    *model_name* selects the model version whose sources are read; it is forwarded only when given.
    """
    if sys.platform not in ("win32", "linux"):
        return None
    from zvisiongenerator.backends.cuda.video.memory_estimate import cached_ltx_system_estimate

    if model_name is None:
        return cached_ltx_system_estimate(config, resolved_model, quantize, find_local_dir=find_local_dir)
    return cached_ltx_system_estimate(config, resolved_model, quantize, find_local_dir=find_local_dir, model_name=model_name)


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

    macOS writes an MLX folder with an MLX-quantized transformer; Windows and Linux write a diffusers folder with an
    NF4 or FP8 transformer.

    Raises:
        RuntimeError: On another platform, or when the conversion fails.
    """
    if sys.platform == "darwin":
        from zvisiongenerator.backends.mac.ltx_converted import convert as convert_mlx

        convert_mlx(checkpoint, target, bits, config=config, models_dir=models_dir, lora=lora, report=report)
        return
    if sys.platform in ("win32", "linux"):
        from zvisiongenerator.backends.cuda.video.converted import convert

        convert(checkpoint, target, bits, config=config, models_dir=models_dir, lora=lora, report=report)
        return
    raise RuntimeError(f"Unsupported platform: {sys.platform}. Converting LTX checkpoints supports macOS, Windows, and Linux.")


# --- Download plans ---
# Each platform has a light planner module (no torch, diffusers or mflux import) that lists the Hugging Face
# downloads a model needs, so a plan is available without a backend instance.


def image_download_plan(model_ref: str, *, family: str, read_index: Callable[[str, str | None], dict[str, Any]] | None = None) -> "tuple[DownloadTarget, ...]":
    """Return the downloads this platform's image backend needs for *model_ref* (empty for a local folder).

    Args:
        model_ref: A local path or a Hugging Face ``owner/repo[@revision]`` reference.
        family: The detected model family.
        read_index: Reads a repo's ``model_index.json`` as ``(repo_id, revision) -> dict`` (diffusers platforms only; defaults to the Hub).

    Raises:
        LookupError: If a diffusers repo's ``model_index.json`` cannot be read.
        RuntimeError: On an unsupported platform.
    """
    if sys.platform == "darwin":
        from zvisiongenerator.backends.mac.download_plan import image_download_plan as plan

        return plan(model_ref, family=family)
    if sys.platform in ("win32", "linux"):
        from zvisiongenerator.backends.cuda.download_plan import image_download_plan as plan

        return plan(model_ref, family=family, read_index=read_index)
    raise RuntimeError(f"Unsupported platform: {sys.platform}. Image generation supports macOS, Windows, and Linux.")


def video_download_plan(
    model_ref: str,
    *,
    family: str,
    transformer_source: tuple[str, str] | None = None,
    latent_upsampler: tuple[str, str] | None = None,
    upscale: bool = False,
    repo_ignore: Sequence[str] = (),
    text_encoder_bundled: bool = False,
) -> "tuple[DownloadTarget, ...]":
    """Return the downloads this platform's video backend needs for *model_ref* (empty for a local folder).

    macOS lists the model repo at its revision and, unless *text_encoder_bundled* (a model whose pack carries its own
    text encoder), the Gemma 3 text encoder; Windows and Linux list the diffusers repo (without the
    *repo_ignore* patterns), the single-file *transformer_source* and, with *upscale*, the *latent_upsampler*
    (each a ``(repo_id[@revision], filename)`` pair).

    Raises:
        RuntimeError: On an unsupported platform.
    """
    if sys.platform == "darwin":
        from zvisiongenerator.backends.mac.download_plan import video_download_plan as plan
    elif sys.platform in ("win32", "linux"):
        from zvisiongenerator.backends.cuda.download_plan import video_download_plan as plan
    else:
        raise RuntimeError(f"Unsupported video platform: {sys.platform}. Video generation supports macOS, Windows, and Linux.")
    return plan(
        model_ref,
        family=family,
        transformer_source=transformer_source,
        latent_upsampler=latent_upsampler,
        upscale=upscale,
        repo_ignore=repo_ignore,
        text_encoder_bundled=text_encoder_bundled,
    )


def h3_download_plan(model_ref: str, preset: "H3Preset | H3MlxPreset", *, upscale: bool = False, lora: "HubFiles | None" = None) -> "tuple[DownloadTarget, ...]":
    """Return the downloads this platform's MiniMax H3 backend needs for *model_ref*.

    Windows and Linux list the repository, the text encoder and the turbo LoRA, plus the latent upscaler with *upscale*. macOS lists
    the INT8 transformer, the ``FL2VA`` components and the turbo LoRA *lora*; it has no upscale.

    Raises:
        RuntimeError: On a platform without an H3 backend.
        ValueError: On macOS without *lora*.
    """
    if sys.platform == "darwin":
        from zvisiongenerator.backends.mac.download_plan import h3_mlx_download_plan

        if lora is None:
            raise ValueError("The MiniMax H3 download plan on macOS needs the turbo LoRA (lora=).")
        return h3_mlx_download_plan(model_ref, cast("H3MlxPreset", preset), lora)
    if sys.platform in ("win32", "linux"):
        from zvisiongenerator.backends.cuda.download_plan import h3_download_plan as plan

        return plan(model_ref, cast("H3Preset", preset), upscale=upscale)
    raise RuntimeError(f"MiniMax H3 is not available on this platform ({sys.platform}).")


def h3_companion_repo_ids(config: dict[str, Any]) -> tuple[str, ...]:
    """Return the Hugging Face repos, besides the main one, that this platform's MiniMax H3 backend downloads.

    Windows and Linux download the text encoder, the turbo LoRA and the latent upscaler; macOS downloads the INT8 transformer and the turbo LoRA.

    Raises:
        ValueError: If the H3 entry of *config* is malformed.
    """
    if sys.platform == "darwin":
        from zvisiongenerator.utils.h3_preset import h3_mlx_preset, h3_turbo_lora

        return (h3_mlx_preset(config).transformer.repo_id, h3_turbo_lora(config)[0].repo_id)
    from zvisiongenerator.utils.h3_preset import h3_preset

    preset = h3_preset(config)
    return (preset.text_encoder.repo_id, preset.lora.repo_id, preset.latent_upsampler.repo_id)


def h3_load_options(config: dict[str, Any], model_name: str | None, *, quiet: bool = False) -> dict[str, Any]:
    """Return the keyword arguments this platform's MiniMax H3 backend takes in ``load_model`` beyond the mode and LoRAs.

    Windows and Linux take the model name and the CUDA preset. macOS takes the model name, the MLX preset and the turbo
    LoRA; with *quiet* (a run with no terminal, such as the web runner) the fork's progress output is off.

    Raises:
        ValueError: If the H3 entry of *config* is malformed.
    """
    if sys.platform == "darwin":
        from zvisiongenerator.utils.h3_preset import h3_mlx_load_options

        return {**h3_mlx_load_options(config, model_name), **({"verbose": False} if quiet else {})}
    from zvisiongenerator.utils.h3_preset import h3_load_options as cuda_load_options

    return cuda_load_options(config, model_name)


def enhancer_download_plan(repo: str, revision: str | None) -> "tuple[DownloadTarget, ...]":
    """Return the download of the prompt-enhancer model *repo* for this platform's loader (empty for a local folder).

    Raises:
        RuntimeError: On an unsupported platform.
    """
    if sys.platform == "darwin":
        from zvisiongenerator.backends.mac.download_plan import enhancer_download_plan as plan
    elif sys.platform in ("win32", "linux"):
        from zvisiongenerator.backends.cuda.download_plan import enhancer_download_plan as plan
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}. Prompt enhancement supports macOS, Windows, and Linux.")
    return plan(repo, revision)


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
