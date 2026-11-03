"""The audio rows of the Models page: one per audio model that runs on this machine, with its downloads, disk use and delete."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.audio_preset import AudioPreset, parse_audio_presets
from zvisiongenerator.utils.hf_cache import repo_disk_bytes
from zvisiongenerator.utils.model_files import disk_size_label, huggingface_cache_repo_dir
from zvisiongenerator.utils.model_memory import MemoryBudget
from zvisiongenerator.web.audio_models import audio_rows
from zvisiongenerator.web.model_delete import audio_delete_target

__all__ = ["audio_inventory_rows", "audio_repo_ids"]


def audio_repo_ids(config: Mapping[str, Any]) -> frozenset[str]:
    """Return every repository any audio preset of *config* loads, so the cache listing never offers them as unused."""
    return frozenset(files.repo_id for preset in parse_audio_presets(config) for files in preset.files)


def audio_inventory_rows(
    config: Mapping[str, Any],
    budget: MemoryBudget | None,
    *,
    target_present: Callable[[DownloadTarget], bool] | None = None,
) -> list[dict[str, Any]]:
    """Return one Models-page row per audio model that runs on this machine, in config order; *target_present* defaults to the offline check."""
    return audio_rows(config, budget, _inventory_fields, target_present=target_present)


def _inventory_fields(preset: AudioPreset, shared: dict[str, Any]) -> dict[str, Any]:
    """Return the keys only the Models-page row has: the name, whether a download is needed and what deleting removes.

    One unreadable model warns and reads as unknown instead of breaking the listing.
    """
    fields: dict[str, Any] = {"name": preset.id, "download_needed": shared["downloaded"] is False}
    try:
        target = audio_delete_target(preset)
        return {**fields, "delete": _delete_info(target), "disk_size_label": _disk_size_label(target)}
    except Exception as exc:  # noqa: BLE001 - one unreadable model must not break the whole listing
        warnings.warn(f"Could not describe audio model '{preset.id}': {exc}", stacklevel=2)
        return {**fields, "delete": None, "disk_size_label": None}


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
