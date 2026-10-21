"""Load SeedVR2 from its weights folder, filling the folder the way the platform engine keeps its weights.

Each tier has a hidden folder at ``<models dir>/.seedvr2/<model>@<precision>``.

On macOS the first load lets mflux download the tier's 16-bit files from Hugging Face and quantize them to 8 bits
while loading. The quantized weights are then saved to the folder as a copy and the two downloaded files are
deleted, so a tier occupies the disk once, at its 8-bit size. Later loads read the copy.

On Windows and Linux the Hub's FP8 files are already 8-bit, and the vendored core reads the transformer and the
VAE from one folder, so the engine downloads both files into the tier's folder and reads them from there.
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
    SEEDVR2_REPO,
    SeedVR2Setup,
    SeedVR2Weights,
    pick_seedvr2_tier,
    seedvr2_row_name,
    seedvr2_settings,
    seedvr2_tier_label,
)
from zvisiongenerator.utils.stored_quant import manifest_matches

__all__ = [
    "DOWNLOADING_PHASE",
    "LOADING_PHASE",
    "SAVING_QUANT_PHASE",
    "SEEDVR2_STORED_FORMAT",
    "build_seedvr2_setup",
    "load_seedvr2",
    "seedvr2_downloaded",
    "seedvr2_weights_dir",
]

SEEDVR2_STORED_FORMAT = "mflux-seedvr2"
# Reported while the tier's files are fetched (and, on macOS, quantized), before they are usable from disk.
DOWNLOADING_PHASE = "downloading"
_WEIGHTS_DIR_NAME = ".seedvr2"


def seedvr2_weights_dir(model: str, models_dir: Path, weights: SeedVR2Weights) -> Path:
    """Return the folder that holds *model*'s weights as *weights* keep them; it is hidden from the model listings."""
    return models_dir / _WEIGHTS_DIR_NAME / seedvr2_row_name(model, weights)


def seedvr2_downloaded(model: str, models_dir: Path, weights: SeedVR2Weights) -> bool:
    """Return whether *model* loads without a download: its current stored copy, or both of its downloaded files."""
    folder = seedvr2_weights_dir(model, models_dir, weights)
    if weights.stores_copy:
        return manifest_matches(folder, _manifest(model))
    return all((folder / name).is_file() for _repo, name in weights.files[model])


def build_seedvr2_setup(config: dict[str, Any], *, weights: SeedVR2Weights, budget_bytes: int | None, system_bytes: int | None, models_dir: Path) -> SeedVR2Setup:
    """Return the tier a plain upscale uses with *config* on this machine, and whether its weights are on disk.

    Args:
        config: Loaded config dict.
        weights: The engine's weights; they name the config tier list that applies.
        budget_bytes: The accelerator memory budget, or ``None`` when unknown.
        system_bytes: The system memory weights stream from on a discrete GPU, or ``None`` (Apple Silicon, unknown).
        models_dir: The installed-models directory.

    Raises:
        ValueError: If the ``seedvr2`` block of *config* is invalid.
    """
    settings = seedvr2_settings(config, weights.tiers_key)
    tier = pick_seedvr2_tier(settings, budget_bytes, system_bytes)
    return SeedVR2Setup(settings=settings, tier=tier, budget_bytes=budget_bytes, downloaded=seedvr2_downloaded(tier.model, models_dir, weights), weights=weights)


def _delete_cached_file(repo_id: str, filename: str) -> None:
    """Delete *filename* of *repo_id* from the Hugging Face cache: its snapshot link, then the blob it points to."""
    from huggingface_hub import try_to_load_from_cache

    link = try_to_load_from_cache(repo_id, filename)
    if not isinstance(link, str):
        return
    blob = Path(link).resolve()
    Path(link).unlink(missing_ok=True)
    blob.unlink(missing_ok=True)


def _cached_file(repo_id: str, filename: str) -> object:
    """Look up *filename* of *repo_id* in the Hugging Face cache; a path string means it is cached."""
    from huggingface_hub import try_to_load_from_cache

    return try_to_load_from_cache(repo_id, filename)


def _files_on_disk(files: tuple[tuple[str, str], ...], files_cached: Callable[[str, str], object]) -> bool:
    """Return whether every ``(repo, name)`` of *files* is in the Hugging Face cache."""
    return all(isinstance(files_cached(repo, name), str) for repo, name in files)


def load_seedvr2(
    upscaler: SeedVR2Upscaler,
    model: str,
    *,
    models_dir: Path,
    on_phase: Callable[[str], None] | None,
    delete_cached_file: Callable[[str, str], None] = _delete_cached_file,
    files_cached: Callable[[str, str], object] = _cached_file,
) -> Any:
    """Load *model* at 8 bits from its weights folder, filling the folder on first use.

    An engine that stores a copy loads it, or on first use creates it from the download and then frees the
    download. Any other engine is given the folder and downloads the files it is missing into it.

    Args:
        upscaler: The platform SeedVR2 engine.
        model: "seedvr2-3b" or "seedvr2-7b".
        models_dir: The installed-models directory; the weights live in its hidden ``.seedvr2`` folder.
        on_phase: Receives :data:`LOADING_PHASE` when every file the load reads is on disk (a stored copy, the
            downloaded files, or, where a copy is stored, the tier's files in the Hugging Face cache), otherwise
            :data:`DOWNLOADING_PHASE` while the library fetches them, then (for a stored copy)
            :data:`SAVING_QUANT_PHASE` while it is written. A download the job ran beforehand reports its own
            progress, so the load then reports :data:`LOADING_PHASE`.
        delete_cached_file: Removes one file of a Hugging Face repo from the local cache; the default deletes the
            cache link and its blob.
        files_cached: Looks one file up in the Hugging Face cache, returning its path when it is cached; the default
            is ``huggingface_hub.try_to_load_from_cache``.

    Returns:
        The loaded model.
    """
    weights = upscaler.weights
    folder = seedvr2_weights_dir(model, models_dir, weights)
    if seedvr2_downloaded(model, models_dir, weights):
        _report(on_phase, LOADING_PHASE)
        return upscaler.load_model(model, quantize=None, model_path=str(folder))

    on_disk = weights.stores_copy and _files_on_disk(weights.files[model], files_cached)
    _report(on_phase, LOADING_PHASE if on_disk else DOWNLOADING_PHASE)
    if not weights.stores_copy:
        folder.mkdir(parents=True, exist_ok=True)
        return upscaler.load_model(model, quantize=None, model_path=str(folder))

    # mflux fetches any missing file, and the load keeps the quantized weights in memory, so nothing reads the
    # downloaded files again and they can be deleted once the copy is stored.
    loaded = upscaler.load_model(model, quantize=SEEDVR2_BITS, model_path=SEEDVR2_REPO)
    _report(on_phase, SAVING_QUANT_PHASE)
    folder.parent.mkdir(parents=True, exist_ok=True)
    saved = store_copy(
        lambda partial: upscaler.save_model(loaded, partial), target=folder, manifest=_manifest(model), label=seedvr2_tier_label(model, weights), fallback="the next load saves it again"
    )
    if saved:
        _delete_downloads(weights.files[model], delete_cached_file)
    return loaded


def _manifest(model: str) -> dict[str, Any]:
    return {"bits": SEEDVR2_BITS, "backend_format": SEEDVR2_STORED_FORMAT, "model": model}


def _report(on_phase: Callable[[str], None] | None, phase: str) -> None:
    if on_phase is not None:
        on_phase(phase)


def _delete_downloads(files: tuple[tuple[str, str], ...], delete_cached_file: Callable[[str, str], None]) -> None:
    """Delete the model's Hugging Face *files*; one that cannot be deleted only warns, as the copy is already stored."""
    for repo, name in files:
        try:
            delete_cached_file(repo, name)
        except OSError as exc:
            warnings.warn(f"Could not delete the downloaded {name} ({exc}); it stays in the Hugging Face cache.", stacklevel=3)
