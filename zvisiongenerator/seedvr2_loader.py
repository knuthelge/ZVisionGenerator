"""Load SeedVR2, reusing its stored 8-bit copy or creating it on first use.

The first load lets mflux download the tier's 16-bit files from Hugging Face and quantize them to 8 bits while
loading. The quantized weights are then saved as a hidden copy at ``<models dir>/.seedvr2/<model>@q8`` and the
two downloaded files are deleted, so a tier occupies the disk once, at its 8-bit size. Later loads read the copy.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zvisiongenerator.core.seedvr2 import SeedVR2Upscaler
from zvisiongenerator.image_model_loader import LOADING_PHASE, SAVING_QUANT_PHASE, store_copy
from zvisiongenerator.utils.seedvr2 import (
    SEEDVR2_BITS,
    SEEDVR2_MODEL_FILES,
    SEEDVR2_REPO,
    SeedVR2Setup,
    pick_seedvr2_tier,
    seedvr2_settings,
    seedvr2_tier_label,
)
from zvisiongenerator.utils.stored_quant import manifest_matches, stored_quant_name

__all__ = [
    "DOWNLOADING_PHASE",
    "LOADING_PHASE",
    "SAVING_QUANT_PHASE",
    "SEEDVR2_STORED_FORMAT",
    "build_seedvr2_setup",
    "load_seedvr2",
    "seedvr2_copy_dir",
    "seedvr2_downloaded",
]

SEEDVR2_STORED_FORMAT = "mflux-seedvr2"
# Reported while mflux fetches the tier's files, before the copy exists.
DOWNLOADING_PHASE = "downloading"
_COPY_DIR_NAME = ".seedvr2"


def seedvr2_copy_dir(model: str, models_dir: Path) -> Path:
    """Return where *model*'s stored 8-bit copy lives; the folder is hidden from the model listings."""
    return models_dir / _COPY_DIR_NAME / stored_quant_name(model, SEEDVR2_BITS)


def seedvr2_downloaded(model: str, models_dir: Path) -> bool:
    """Return whether a current stored copy of *model* exists."""
    return manifest_matches(seedvr2_copy_dir(model, models_dir), _manifest(model))


def build_seedvr2_setup(config: dict[str, Any], *, budget_bytes: int | None, models_dir: Path) -> SeedVR2Setup:
    """Return the tier a plain upscale uses with *config* on a Mac with *budget_bytes*, and whether its copy is on disk.

    Raises:
        ValueError: If the ``seedvr2`` block of *config* is invalid.
    """
    settings = seedvr2_settings(config)
    tier = pick_seedvr2_tier(settings, budget_bytes)
    return SeedVR2Setup(settings=settings, tier=tier, budget_bytes=budget_bytes, downloaded=seedvr2_downloaded(tier.model, models_dir))


def _delete_cached_file(repo_id: str, filename: str) -> None:
    """Delete *filename* of *repo_id* from the Hugging Face cache: its snapshot link, then the blob it points to."""
    from huggingface_hub import try_to_load_from_cache

    link = try_to_load_from_cache(repo_id, filename)
    if not isinstance(link, str):
        return
    blob = Path(link).resolve()
    Path(link).unlink(missing_ok=True)
    blob.unlink(missing_ok=True)


def load_seedvr2(
    upscaler: SeedVR2Upscaler,
    model: str,
    *,
    models_dir: Path,
    on_phase: Callable[[str], None] | None,
    delete_cached_file: Callable[[str, str], None] = _delete_cached_file,
) -> Any:
    """Load *model* at 8 bits from its stored copy, creating the copy and freeing the download on first use.

    Args:
        upscaler: The platform SeedVR2 engine.
        model: "seedvr2-3b" or "seedvr2-7b".
        models_dir: The installed-models directory; the copy lives in its hidden ``.seedvr2`` folder.
        on_phase: Receives :data:`LOADING_PHASE` when a stored copy is read, otherwise :data:`DOWNLOADING_PHASE`
            while the files are fetched and quantized, then :data:`SAVING_QUANT_PHASE` while the copy is written.
        delete_cached_file: Removes one file of a Hugging Face repo from the local cache; the default deletes the
            cache link and its blob.

    Returns:
        The loaded model, its weights resident in memory.
    """
    copy = seedvr2_copy_dir(model, models_dir)
    if seedvr2_downloaded(model, models_dir):
        _report(on_phase, LOADING_PHASE)
        return upscaler.load_model(model, quantize=None, model_path=str(copy))

    _report(on_phase, DOWNLOADING_PHASE)
    # mflux fetches any missing file, and the load keeps the quantized weights in memory, so nothing reads the
    # downloaded files again and they can be deleted once the copy is stored.
    loaded = upscaler.load_model(model, quantize=SEEDVR2_BITS, model_path=SEEDVR2_REPO)
    _report(on_phase, SAVING_QUANT_PHASE)
    copy.parent.mkdir(parents=True, exist_ok=True)
    saved = store_copy(lambda partial: upscaler.save_model(loaded, partial), target=copy, manifest=_manifest(model), label=seedvr2_tier_label(model))
    if saved:
        _delete_downloads(model, delete_cached_file)
    return loaded


def _manifest(model: str) -> dict[str, Any]:
    return {"bits": SEEDVR2_BITS, "backend_format": SEEDVR2_STORED_FORMAT, "model": model}


def _report(on_phase: Callable[[str], None] | None, phase: str) -> None:
    if on_phase is not None:
        on_phase(phase)


def _delete_downloads(model: str, delete_cached_file: Callable[[str, str], None]) -> None:
    """Delete *model*'s two Hugging Face files; one that cannot be deleted only warns, as the copy is already stored."""
    for name in SEEDVR2_MODEL_FILES[model]:
        try:
            delete_cached_file(SEEDVR2_REPO, name)
        except OSError as exc:
            warnings.warn(f"Could not delete the downloaded {name} ({exc}); it stays in the Hugging Face cache.", stacklevel=3)
