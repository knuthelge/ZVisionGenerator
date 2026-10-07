"""Describe whether a model is downloaded and whether it fits this machine's memory."""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zvisiongenerator.backends import get_accelerator_memory_budget
from zvisiongenerator.utils.model_files import find_local_model_dir, is_ltx_mlx_layout, ltx_mlx_text_encoder_repo
from zvisiongenerator.utils.model_memory import GIB, classify_memory_fit, estimate_image_memory, estimate_ltx_mlx_memory

NO_QUANTIZE_KEY = "none"
VIDEO_ESTIMATE_NOTE = "Excludes the optional upscale pass, which needs more."


def describe_model_status(
    resolved_path: str,
    *,
    kind: str,
    quantize_options: tuple[int, ...] = (),
    budget_bytes: int | None = None,
    find_local_dir: Callable[[str], Path | None] = find_local_model_dir,
) -> dict[str, Any]:
    """Return the ``downloaded`` flag and ``memory_fit`` summary for one inventory model.

    Args:
        resolved_path: The model's local directory or HuggingFace repo id.
        kind: ``"image"`` or ``"video"``.
        quantize_options: Quantization levels the model supports (empty when it cannot be quantized).
        budget_bytes: GPU memory budget, or ``None`` when this platform is not estimated.
        find_local_dir: Resolver for fully downloaded model directories.

    Returns:
        ``{"downloaded": bool, "memory_fit": dict | None}``. ``memory_fit`` holds ``budget_gb``, ``by_quantize``
        (``"none"`` and each quantize level mapped to ``{"status", "required_gb"}``), and for LTX MLX video a
        ``without_low_memory`` estimate in the same shape plus a ``note``. It is ``None`` when the model cannot
        be estimated.
    """
    model_dir = find_local_dir(resolved_path)
    text_encoder_dir = None
    if kind == "video" and model_dir is not None and is_ltx_mlx_layout(model_dir):
        text_encoder_dir = find_local_dir(ltx_mlx_text_encoder_repo())
        downloaded = text_encoder_dir is not None
    else:
        downloaded = model_dir is not None

    memory_fit = None
    if downloaded and model_dir is not None and budget_bytes:
        if kind == "video":
            memory_fit = _video_memory_fit(model_dir, text_encoder_dir, budget_bytes)
        else:
            levels = estimate_image_memory(model_dir, (None, *quantize_options))
            if levels is not None:
                by_quantize = {NO_QUANTIZE_KEY if level is None else str(level): _fit(value, budget_bytes) for level, value in levels.items()}
                memory_fit = {"budget_gb": _to_gb(budget_bytes), "by_quantize": by_quantize}
    return {"downloaded": downloaded, "memory_fit": memory_fit}


@functools.cache
def memory_budget_bytes() -> int | None:
    """Return the accelerator memory budget, read once per process (it is fixed hardware)."""
    try:
        return get_accelerator_memory_budget()
    except Exception:  # noqa: BLE001 - a missing or broken MLX install simply disables the badge
        return None


def _video_memory_fit(model_dir: Path, text_encoder_dir: Path | None, budget_bytes: int) -> dict[str, Any] | None:
    if text_encoder_dir is None:
        return None
    staged = estimate_ltx_mlx_memory(model_dir, text_encoder_dir, low_memory=True)
    resident = estimate_ltx_mlx_memory(model_dir, text_encoder_dir, low_memory=False)
    if staged is None or resident is None:
        return None
    return {
        "budget_gb": _to_gb(budget_bytes),
        "by_quantize": {NO_QUANTIZE_KEY: _fit(staged, budget_bytes)},
        "without_low_memory": _fit(resident, budget_bytes),
        "note": VIDEO_ESTIMATE_NOTE,
    }


def _fit(required_bytes: int, budget_bytes: int) -> dict[str, Any]:
    return {"status": classify_memory_fit(required_bytes, budget_bytes), "required_gb": _to_gb(required_bytes)}


def _to_gb(value: int) -> float:
    return round(value / GIB, 1)
