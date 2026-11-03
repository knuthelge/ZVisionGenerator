"""The audio rows of the Models page: one per audio model that runs on this machine, with its downloads, disk use and delete."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from zvisiongenerator.downloads.presence import target_present_offline
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.audio_preset import AudioPreset, parse_audio_presets
from zvisiongenerator.utils.hf_cache import repo_disk_bytes
from zvisiongenerator.utils.licences import LicenceRecord, licence_payload, load_licences
from zvisiongenerator.utils.model_files import disk_size_label, huggingface_cache_repo_dir
from zvisiongenerator.utils.model_memory import MemoryBudget
from zvisiongenerator.web.audio_models import runnable_audio_models
from zvisiongenerator.web.model_delete import audio_delete_target
from zvisiongenerator.web.model_download_plan import audio_download_needed
from zvisiongenerator.web.model_status import describe_audio_status

__all__ = ["audio_inventory_rows", "audio_repo_ids"]


def audio_repo_ids(config: Mapping[str, Any]) -> frozenset[str]:
    """Return every repository any audio preset of *config* loads on any platform, so the cache listing never offers them as unused."""
    try:
        presets = parse_audio_presets(config)
    except ValueError as exc:
        warnings.warn(f"Could not read the audio model presets: {exc}", stacklevel=2)
        return frozenset()
    return frozenset(files.repo_id for preset in presets for platform in preset.platforms.values() for files in platform.files)


def audio_inventory_rows(
    config: Mapping[str, Any],
    budget: MemoryBudget | None,
    *,
    target_present: Callable[[DownloadTarget], bool] | None = None,
) -> list[dict[str, Any]]:
    """Return one Models-page row per audio model that runs on this machine, in config order.

    A row holds ``name`` (the model id), ``label``, ``kind`` (``music``, ``speech`` or ``sound``), ``licences``,
    ``downloaded``, ``download_needed``, ``memory_fit`` (the GPU fit only; see :func:`describe_audio_status`),
    ``delete`` (what deleting it removes, or ``None`` with nothing on disk) and ``disk_size_label``. A model is
    ``downloaded`` when every repository of its plan is on disk.

    Args:
        config: The loaded config.
        budget: This machine's memory budget, or ``None`` when it is not estimated.
        target_present: Tells whether one download target is fully on disk; the offline cache check by default.
    """
    presets, platform_key = runnable_audio_models(config)
    if platform_key is None:
        return []
    present = target_present or target_present_offline
    licences = load_licences()
    return [_row(licences, preset, platform_key, budget, present) for preset in presets]


def _row(licences: Sequence[LicenceRecord], preset: AudioPreset, platform_key: str, budget: MemoryBudget | None, present: Callable[[DownloadTarget], bool]) -> dict[str, Any]:
    """Build the Models-page row of one audio model; one unreadable model warns and reads as unknown instead of breaking the listing."""
    row: dict[str, Any] = {"name": preset.id, "label": preset.label, "kind": preset.kind, "licences": licence_payload(licences, preset.id)}
    try:
        status = describe_audio_status(preset, platform_key, budget=budget, target_present=present)
        needed = audio_download_needed(preset, platform_key, present=present)
        target = audio_delete_target(preset, platform_key)
        return {
            **row,
            "downloaded": status["downloaded"],
            "download_needed": needed,
            "memory_fit": status["memory_fit"],
            "delete": _delete_info(target),
            "disk_size_label": _disk_size_label(target),
        }
    except Exception as exc:  # noqa: BLE001 - one unreadable model must not break the whole listing
        warnings.warn(f"Could not describe audio model '{preset.id}': {exc}", stacklevel=2)
        return {**row, "downloaded": None, "download_needed": False, "memory_fit": None, "delete": None, "disk_size_label": None}


def _delete_info(target: Any) -> dict[str, Any] | None:
    """Return the row's ``delete`` field: what deleting removes, in the shape of an image model's."""
    if target is None or not target.on_disk:
        return None
    return {"kind": target.kind, "repo_id": target.repo_id, "linked_by": [], "stored_quants": [], "source_repos": []}


def _disk_size_label(target: Any) -> str | None:
    """Return the disk the model's repositories use, or ``None`` when nothing is on disk.

    That is what deleting the model frees: the repositories of its delete target, which share no file today.
    """
    if target is None:
        return None
    total = sum(repo_disk_bytes(huggingface_cache_repo_dir(repo_id)) for repo_id in target.repo_ids)
    return disk_size_label(total) if total else None
