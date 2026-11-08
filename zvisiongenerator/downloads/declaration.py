"""Read the downloads a built-in model declares in the packaged metadata, as download targets.

A model's declaration is its own ``downloads`` followed by the ``companions`` of its video family (overlaid by its version),
each target paired with its role. This module only reads: the planners decide which of the declared targets a run needs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.model_metadata import family_block, for_platform, model_entry
from zvisiongenerator.utils.paths import split_huggingface_revision

__all__ = ["companion_targets", "declared_repo_ids", "model_targets", "role_targets"]

_DEFAULT_ROLE = "model"
_VIDEO_KIND = "video"


def model_targets(model_id: str, platform_key: str) -> tuple[tuple[str, DownloadTarget], ...] | None:
    """Return the ``(role, target)`` pairs the built-in model *model_id* downloads on *platform_key*, or ``None`` when it declares none there.

    The pairs are the model's own targets in file order, then its family's companions in file order. ``None`` means the
    model has no entry or its entry has no downloads for the platform; an empty tuple means it declares nothing to fetch.
    A target listed with ``files_from: loader`` has no file patterns (``allow_patterns`` is ``None``): it is used for its repo
    and revision only.
    """
    entry = model_entry(model_id)
    if entry is None:
        return None
    own = for_platform(entry.get("downloads"), platform_key)
    if own is None:
        return None
    pairs = tuple(_pair(target) for target in own)
    if entry["kind"] == _VIDEO_KIND:
        pairs += companion_targets(entry["family"], entry.get("version"), platform_key)
    return pairs


def companion_targets(family: str, version: str | None, platform_key: str) -> tuple[tuple[str, DownloadTarget], ...]:
    """Return the ``(role, target)`` pairs every model of the video *family* (at *version*) also downloads on *platform_key*.

    Raises:
        KeyError: If the metadata has no such family or version.
    """
    companions = for_platform(family_block(family, version).get("companions"), platform_key)
    return tuple(_pair(target) for target in companions or ())


def role_targets(pairs: Iterable[tuple[str, DownloadTarget]], role: str) -> tuple[DownloadTarget, ...]:
    """Return the targets of *pairs* that have *role*, in order."""
    return tuple(target for target_role, target in pairs if target_role == role)


def declared_repo_ids(model_id: str, platform_key: str) -> tuple[str, ...]:
    """Return the repo ids the built-in model *model_id* declares on *platform_key*, in order and without repeats; empty when it declares none."""
    pairs = model_targets(model_id, platform_key) or ()
    return tuple(dict.fromkeys(target.repo_id for _, target in pairs))


def _pair(spec: Mapping[str, Any]) -> tuple[str, DownloadTarget]:
    """Return the role and the frozen target a metadata target mapping describes."""
    repo_id, revision = split_huggingface_revision(spec["ref"])
    files = spec.get("files")
    ignore = spec.get("ignore")
    target = DownloadTarget(
        repo_id,
        revision,
        allow_patterns=None if files is None else tuple(files),
        ignore_patterns=None if ignore is None else tuple(ignore),
    )
    return spec.get("role", _DEFAULT_ROLE), target
