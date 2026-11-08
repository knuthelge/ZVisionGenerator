"""Read the facts about the built-in prompt-enhancer models (Fast and Smart) from the packaged model metadata."""

from __future__ import annotations

from zvisiongenerator.downloads.declaration import model_targets
from zvisiongenerator.utils.model_metadata import for_platform, model_entry

__all__ = ["ENHANCER_TIERS", "enhancer_model_id", "enhancer_tier_models", "smart_unavailable"]

ENHANCER_TIERS = ("fast", "smart")
_GIB = 2**30


def enhancer_model_id(tier: str) -> str:
    """Return the id of the ``models`` entry of the prompt enhancer *tier*, which is also the model id of its licence records (``enhancer-fast``, ``enhancer-smart``)."""
    return f"enhancer-{tier}"


def enhancer_tier_models(*, platform_key: str) -> dict[str, tuple[str, str | None]]:
    """Return the built-in enhancer ``(repo, revision)`` per tier, ``fast`` and ``smart``, as the metadata declares them for *platform_key*."""
    models = {}
    for tier in ENHANCER_TIERS:
        target = model_targets(enhancer_model_id(tier), platform_key)[0][1]
        models[tier] = (target.repo_id, target.revision)
    return models


def smart_unavailable(*, platform_key: str, budget_bytes: int | None) -> str | None:
    """Return why the Smart enhancer cannot run on this machine, or None when it can.

    Args:
        platform_key: Platform key such as ``darwin`` or ``win32``.
        budget_bytes: The accelerator memory (VRAM, or the GPU working set on Apple Silicon); ``None`` when unknown,
            which counts as unavailable.
    """
    if budget_bytes is None:
        return "Smart is not available on this machine."
    need = for_platform(model_entry(enhancer_model_id("smart"))["memory_gb"], platform_key)
    if budget_bytes < need * _GIB:
        return f"Needs about {need:g} GB of memory; this machine has {budget_bytes / _GIB:.0f} GB."
    return None
