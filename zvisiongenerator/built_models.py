"""One entry point for the models built on first use (Ideogram 4 Instant, MOSS-Image).

Callers ask here to prepare a model, so none of them names a family. Each family's build specifics live in its own
builder module, which is imported only when a build is needed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from zvisiongenerator.utils.ideogram4_instant import is_instant_dir
from zvisiongenerator.utils.moss_build import is_moss_dir

__all__ = ["ensure_built_model"]


def ensure_built_model(
    model_path: str,
    config: dict[str, Any],
    *,
    platform_key: str,
    on_phase: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> bool:
    """Build the model at *model_path* unless it is already built from the configured sources.

    Any other *model_path* is left alone.

    Args:
        model_path: The resolved model path.
        config: The app config, whose source blocks pin what the model is built from.
        platform_key: Picks the platform's sources (``sys.platform``).
        on_phase: Receives ``BUILDING_PHASE`` when a build starts, then ``LOADING_PHASE`` once it is done.
        cancelled: Checked during the build; a true result stops it.

    Returns:
        Whether the model is ready (``False`` only when *cancelled* stopped the build).

    Raises:
        ValueError: If the model's source block in *config* is invalid.
    """
    if is_instant_dir(model_path):
        from zvisiongenerator.ideogram4_instant_builder import ensure_instant_model

        return ensure_instant_model(model_path, config, platform_key=platform_key, on_phase=on_phase, cancelled=cancelled)
    if is_moss_dir(model_path):
        from zvisiongenerator.moss_builder import ensure_moss_model

        return ensure_moss_model(model_path, config, on_phase=on_phase, cancelled=cancelled)
    return True
