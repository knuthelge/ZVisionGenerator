"""Describe whether a model is downloaded and whether it fits this machine's memory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from zvisiongenerator.downloads.presence import target_present_offline
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.config import ltx_mlx_text_encoder_repo_id
from zvisiongenerator.utils.h3_memory import estimate_h3_cuda_clip_memory, h3_mlx_clip_fit
from zvisiongenerator.utils.model_files import find_local_model_dir, is_ltx_mlx_layout, ltx_mlx_bundled_text_encoder
from zvisiongenerator.utils.model_memory import (
    GIB,
    CudaMemoryEstimate,
    MemoryBudget,
    classify_cuda_memory_fit,
    classify_ltx_cuda_memory_fit,
    classify_memory_fit,
    estimate_cuda_image_memory,
    estimate_image_memory,
    estimate_ltx_cuda_clip_memory,
    estimate_ltx_mlx_clip_memory,
    ltx_cuda_system_memory,
)

NO_QUANTIZE_KEY = "none"
UNIFIED_MEMORY = "unified"
DISCRETE_MEMORY = "discrete"
VIDEO_ESTIMATE_NOTE = "Estimated for a 5 s clip at 768x448. The Workspace checks the size and length you choose."
H3_ESTIMATE_NOTE = "Estimated for a 5 s clip at 768x448 from the text. The Workspace checks the size and length you choose."
# The clip the badge is estimated for: 5 s at 24 fps, no upscale.
_REFERENCE_CLIP = {"width": 768, "height": 448, "frames": 121}
# The same for MiniMax H3, whose lengths are 17 * n + 5 frames (124 is 5 s), at its default 8 forwards.
_H3_REFERENCE_CLIP = {"width": 768, "height": 448, "frames": 124}
_H3_REFERENCE_FORWARDS = 8


def describe_model_status(
    resolved_path: str,
    *,
    kind: str,
    quantize_options: tuple[int, ...] = (),
    budget: MemoryBudget | None = None,
    find_local_dir: Callable[[str], Path | None] = find_local_model_dir,
    model_name: str | None = None,
    app_config: dict[str, Any],
    family: str | None = None,
    target_present: Callable[[DownloadTarget], bool] = target_present_offline,
) -> dict[str, Any]:
    """Return the ``downloaded`` flag and ``memory_fit`` summary for one inventory model.

    Args:
        resolved_path: The model's local directory or HuggingFace repo id.
        kind: ``"image"`` or ``"video"``.
        quantize_options: Quantization levels the model supports (empty when it cannot be quantized).
        budget: This machine's memory budget, or ``None`` when this platform is not estimated.
        find_local_dir: Resolver for fully downloaded model directories.
        model_name: The name the model is picked by (``ltx-4``); on CUDA it sets a video model's quantize level.
        app_config: The loaded config: it names the Gemma 3 text encoder and the downloads of a video model, and sets a CUDA video model's quantize level.
        family: A video model's family (``"ltx"`` or ``"h3"``); ``None`` is LTX. LTX and H3 have a memory estimate.
        target_present: Tells whether one download target is fully on disk; MiniMax H3 is downloaded when every target of its plan is.

    Returns:
        ``{"downloaded": bool, "memory_fit": dict | None}``. ``memory_fit`` holds ``kind`` (``"unified"`` on
        Apple Silicon, ``"discrete"`` on CUDA), ``budget_gb`` (the GPU budget) and ``by_quantize`` (``"none"``
        and each quantize level mapped to ``{"status", "required_gb"}``, the GPU need). On CUDA it adds
        ``system_budget_gb`` and each estimate's ``system_gb``. LTX MLX video adds a ``without_low_memory``
        estimate in the same shape plus a ``note``; CUDA video has the same, in the ``discrete`` shape. It is
        ``None`` when the model cannot be estimated (its files are not all on disk, or CUDA video has no level).
    """
    model_dir = find_local_dir(resolved_path)
    text_encoder_dir = None
    ltx_video = kind == "video" and family in (None, "ltx")
    h3_video = kind == "video" and family == "h3"
    if h3_video:
        downloaded = _video_downloaded(resolved_path, "h3", model_name, app_config, target_present)
    elif ltx_video and model_dir is not None and is_ltx_mlx_layout(model_dir):
        downloaded = _video_downloaded(resolved_path, "ltx", model_name, app_config, target_present)
        if downloaded and ltx_mlx_bundled_text_encoder(model_dir) is None:  # An LTX-2.5 pack carries its own Gemma 4; the others need the Gemma 3 repo.
            text_encoder_dir = find_local_dir(ltx_mlx_text_encoder_repo_id(app_config))
    else:
        downloaded = model_dir is not None

    memory_fit = None
    if h3_video and downloaded and budget is not None:
        memory_fit = _h3_memory_fit(budget)
    elif downloaded and model_dir is not None and budget is not None and (kind == "image" or ltx_video):
        levels = (None, *quantize_options)
        if budget.system_bytes is not None:
            memory_fit = _cuda_image_memory_fit(model_dir, levels, budget) if kind == "image" else _cuda_video_memory_fit(resolved_path, budget, find_local_dir, model_name, app_config)
        elif kind == "video":
            memory_fit = _video_memory_fit(model_dir, text_encoder_dir, budget)
        else:
            memory_fit = _unified_image_memory_fit(model_dir, levels, budget.gpu_bytes)
    return {"downloaded": downloaded, "memory_fit": memory_fit}


def _video_downloaded(resolved_path: str, family: str, model_name: str | None, app_config: dict[str, Any], target_present: Callable[[DownloadTarget], bool]) -> bool:
    """Return whether every download of the video model's plan on this platform is on disk (a plan that cannot be made counts as not downloaded)."""
    from zvisiongenerator.downloads.plans import video_model_downloaded

    return video_model_downloaded(resolved_path, family=family, config=app_config, model_name=model_name or resolved_path, target_present=target_present)


def _h3_memory_fit(budget: MemoryBudget) -> dict[str, Any] | None:
    """Return MiniMax H3's memory fit for the reference clip: the GPU and system memory on CUDA, unified memory on Apple Silicon."""
    return _h3_cuda_memory_fit(budget) if budget.system_bytes is not None else _h3_mlx_memory_fit(budget)


def _h3_cuda_memory_fit(budget: MemoryBudget) -> dict[str, Any]:
    estimate = estimate_h3_cuda_clip_memory(**_H3_REFERENCE_CLIP)
    return {
        "kind": DISCRETE_MEMORY,
        "budget_gb": _to_gb(budget.gpu_bytes),
        "system_budget_gb": _to_gb(budget.system_bytes or 0),
        "by_quantize": {NO_QUANTIZE_KEY: _cuda_fit(estimate, budget, classify=classify_ltx_cuda_memory_fit)},
        "note": VIDEO_ESTIMATE_NOTE,
    }


def _h3_mlx_memory_fit(budget: MemoryBudget) -> dict[str, Any] | None:
    fit = h3_mlx_clip_fit(**_H3_REFERENCE_CLIP, budget=budget, forwards=_H3_REFERENCE_FORWARDS)
    if fit is None:
        return None
    return {
        "kind": UNIFIED_MEMORY,
        "budget_gb": fit["budget_gb"],
        "by_quantize": {NO_QUANTIZE_KEY: {"status": fit["status"], "required_gb": fit["required_gb"]}},
        "note": H3_ESTIMATE_NOTE,
    }


def _unified_image_memory_fit(model_dir: Path, levels: tuple[int | None, ...], budget_bytes: int) -> dict[str, Any] | None:
    estimates = estimate_image_memory(model_dir, levels)
    if estimates is None:
        return None
    by_quantize = {_level_key(level): _fit(value, budget_bytes) for level, value in estimates.items()}
    return {"kind": UNIFIED_MEMORY, "budget_gb": _to_gb(budget_bytes), "by_quantize": by_quantize}


def _cuda_image_memory_fit(model_dir: Path, levels: tuple[int | None, ...], budget: MemoryBudget) -> dict[str, Any] | None:
    estimates = estimate_cuda_image_memory(model_dir, levels)
    if estimates is None:
        return None
    return {
        "kind": DISCRETE_MEMORY,
        "budget_gb": _to_gb(budget.gpu_bytes),
        "system_budget_gb": _to_gb(budget.system_bytes or 0),
        "by_quantize": {_level_key(level): _cuda_fit(estimate, budget) for level, estimate in estimates.items()},
    }


def _cuda_video_memory_fit(resolved_path: str, budget: MemoryBudget, find_local_dir: Callable[[str], Path | None], model_name: str | None, app_config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return an LTX model's memory fit on a CUDA machine, or ``None`` when its level or files are unknown."""
    system = ltx_cuda_system_memory(resolved_path, model_name, app_config, find_local_dir)
    if system is None:
        return None
    staged = estimate_ltx_cuda_clip_memory(system, low_memory=True, **_REFERENCE_CLIP)
    resident = estimate_ltx_cuda_clip_memory(system, low_memory=False, **_REFERENCE_CLIP)
    return {
        "kind": DISCRETE_MEMORY,
        "budget_gb": _to_gb(budget.gpu_bytes),
        "system_budget_gb": _to_gb(budget.system_bytes or 0),
        "by_quantize": {NO_QUANTIZE_KEY: _cuda_fit(staged, budget, classify=classify_ltx_cuda_memory_fit)},
        "without_low_memory": _cuda_fit(resident, budget, classify=classify_ltx_cuda_memory_fit),
        "note": VIDEO_ESTIMATE_NOTE,
    }


def _video_memory_fit(model_dir: Path, text_encoder_dir: Path | None, budget: MemoryBudget) -> dict[str, Any] | None:
    budget_bytes = budget.gpu_bytes
    staged = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, low_memory=True, unified_bytes=budget.unified_bytes, **_REFERENCE_CLIP)
    resident = estimate_ltx_mlx_clip_memory(model_dir, text_encoder_dir, low_memory=False, unified_bytes=budget.unified_bytes, **_REFERENCE_CLIP)
    if staged is None or resident is None:
        return None
    return {
        "kind": UNIFIED_MEMORY,
        "budget_gb": _to_gb(budget_bytes),
        "by_quantize": {NO_QUANTIZE_KEY: _fit(staged, budget_bytes)},
        "without_low_memory": _fit(resident, budget_bytes),
        "note": VIDEO_ESTIMATE_NOTE,
    }


def _fit(required_bytes: int, budget_bytes: int) -> dict[str, Any]:
    return {"status": classify_memory_fit(required_bytes, budget_bytes), "required_gb": _to_gb(required_bytes)}


def _cuda_fit(estimate: CudaMemoryEstimate, budget: MemoryBudget, *, classify: Callable[[CudaMemoryEstimate, MemoryBudget], str] = classify_cuda_memory_fit) -> dict[str, Any]:
    return {"status": classify(estimate, budget), "required_gb": _to_gb(estimate.gpu_bytes), "system_gb": _to_gb(estimate.system_bytes)}


def _level_key(level: int | None) -> str:
    return NO_QUANTIZE_KEY if level is None else str(level)


def _to_gb(value: int) -> float:
    return round(value / GIB, 1)
