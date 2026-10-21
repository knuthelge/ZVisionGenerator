"""Choose the model the first-run dialog preselects and describe the dialog's state for the SPA."""

from __future__ import annotations

from typing import Any

from zvisiongenerator.utils.model_memory import GIB, MemoryBudget

__all__ = ["first_run_payload", "recommended_first_run_model"]

# A card sold with 32 GB reports a little under 32 GiB of VRAM, so the threshold admits anything within 5% of it.
_MEMORY_TOLERANCE = 0.95


def recommended_first_run_model(app_config: dict[str, Any], budget: MemoryBudget | None, image_options: list[str] | tuple[str, ...]) -> str | None:
    """Return the image model alias the first-run dialog preselects, or ``None`` when neither candidate is listed.

    The large model is for machines with at least ``first_run.min_memory_gb`` of memory: the total unified memory
    on Apple Silicon, the VRAM on a CUDA card. Any other machine, including one whose memory is unknown, gets the
    small model. A pick that is not in *image_options* falls back to the small model, if listed.

    Args:
        app_config: The merged app config with its ``first_run`` block.
        budget: This machine's memory budget, or ``None`` when unknown.
        image_options: The image model aliases the Web UI lists.
    """
    section = app_config.get("first_run")
    if not isinstance(section, dict):
        return None
    large, small = section.get("large_model"), section.get("small_model")
    minimum = section.get("min_memory_gb")
    memory = _machine_memory_bytes(budget)
    wants_large = isinstance(minimum, (int, float)) and memory is not None and memory >= minimum * GIB * _MEMORY_TOLERANCE
    pick = large if wants_large else small
    if isinstance(pick, str) and pick in image_options:
        return pick
    return small if isinstance(small, str) and small in image_options else None


def first_run_payload(app_config: dict[str, Any], budget: MemoryBudget | None, image_options: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Return the workspace payload's ``first_run`` block: whether the dialog was skipped and the preselected model."""
    ui = app_config.get("ui")
    skipped = isinstance(ui, dict) and ui.get("first_run_skipped") is True
    return {"skipped": skipped, "recommended_model": recommended_first_run_model(app_config, budget, image_options)}


def _machine_memory_bytes(budget: MemoryBudget | None) -> int | None:
    """Return the memory that decides the recommendation: unified memory on a Mac, VRAM on CUDA, else ``None``."""
    if budget is None:
        return None
    if budget.unified_bytes:
        return budget.unified_bytes
    return budget.gpu_bytes if budget.system_bytes is not None else None
