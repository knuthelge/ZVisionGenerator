"""How an audio model's measured GPU peak fits this machine, and the words that say so.

One fit is shared by the Web UI badge, the CLI's warning before a download and the out-of-memory message, so they
cannot disagree. The functions are pure: the caller passes the memory budget.
"""

from __future__ import annotations

from typing import Any

from zvisiongenerator.utils.audio_preset import AudioPreset
from zvisiongenerator.utils.model_memory import GIB, CudaMemoryEstimate, MemoryBudget, classify_cuda_memory_fit, classify_memory_fit

__all__ = ["audio_memory_fit", "memory_need_text", "memory_warning_text"]


def audio_memory_fit(preset: AudioPreset, platform_key: str, budget: MemoryBudget | None) -> dict[str, Any] | None:
    """Return how the preset's recorded GPU peak on *platform_key* fits *budget*, or ``None`` without a budget or a measured peak.

    Returns:
        ``{"status", "required_gb", "budget_gb", "kind"}``: ``status`` is ``fits``, ``tight`` or ``too_large``, and ``kind``
        is ``"unified"`` on Apple Silicon (the budget is Apple's recommended GPU working set) or ``"discrete"`` on CUDA
        (the budget is the card's memory).
    """
    peak_gb = preset.memory_gb_for(platform_key)
    if budget is None or peak_gb is None:
        return None
    required = round(peak_gb * GIB)
    if budget.system_bytes is None:
        status, kind = classify_memory_fit(required, budget.gpu_bytes), "unified"
    else:
        status, kind = classify_cuda_memory_fit(CudaMemoryEstimate(required, 0), budget), "discrete"
    return {"status": status, "required_gb": round(required / GIB, 1), "budget_gb": round(budget.gpu_bytes / GIB, 1), "kind": kind}


def memory_need_text(fit: dict[str, Any]) -> str:
    """Return what the model needs against what the machine has, such as ``it needs about 8.2 GB and the GPU has 8 GB``."""
    return f"it needs about {fit['required_gb']:g} GB and the GPU has {fit['budget_gb']:g} GB"


def memory_warning_text(label: str, fit: dict[str, Any] | None) -> str | None:
    """Return the warning for a model that is ``too_large`` or ``tight`` on this machine, or ``None`` when it fits or is unknown."""
    if fit is None or fit["status"] == "fits":
        return None
    outcome = "it is likely to run out of GPU memory" if fit["status"] == "too_large" else "it may run slowly or run out of GPU memory"
    return f"{label}: {memory_need_text(fit)}; {outcome}."
