"""Plan the downloads behind a Models-page row or a job: the targets, whether the row needs any, and what they would fetch."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.downloads.check import check_target
from zvisiongenerator.downloads.plans import built_model_plan, enhancer_plan, matte_plan
from zvisiongenerator.downloads.presence import target_present_offline
from zvisiongenerator.downloads.space import space_need
from zvisiongenerator.utils.config import enhancer_tier_models
from zvisiongenerator.utils.licences import LicenceRecord, licence_payload
from zvisiongenerator.utils.matte_models import MATTE_ROW_NAME
from zvisiongenerator.utils.model_files import huggingface_cache_dir
from zvisiongenerator.utils.seedvr2 import seedvr2_row_name
from zvisiongenerator.web.audio_models import find_audio_model
from zvisiongenerator.web.job_downloads import JobDownloadPart, audio_model_part, enhancer_part, image_model_part, matte_part, seedvr2_tier_part, video_model_part

if TYPE_CHECKING:
    from zvisiongenerator.downloads.space import SpaceNeed
    from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck
    from zvisiongenerator.web.config import WebUiConfig
    from zvisiongenerator.web.model_inventory import ImageInventoryEntry

__all__ = [
    "KIND_AUDIO",
    "KIND_ENHANCER",
    "KIND_MATTE",
    "KIND_MODEL",
    "enhancer_download_needed",
    "matte_download_needed",
    "parts_summary",
    "row_download_needed",
    "row_plan",
]

KIND_MODEL = "model"
KIND_AUDIO = "audio"
KIND_ENHANCER = "enhancer"
KIND_MATTE = "matte"


def row_plan(kind: str, name: str, web_config: WebUiConfig) -> JobDownloadPart:
    """Return the download of the Models-page row *name* of *kind* as one part, the shape a job's plan has.

    *kind* is ``"model"`` (image, video and SeedVR2 rows, by name), ``"audio"`` (an audio model, by id), ``"enhancer"``
    (by tier) or ``"matte"`` (the one Background removal row). The part's targets are ``None`` when the plan cannot be
    made now (no connection to read the model's file list).

    Raises:
        ValueError: If *kind* is unknown or no such row exists.
    """
    if kind == KIND_MODEL:
        return _model_row_plan(name, web_config)
    if kind == KIND_AUDIO:
        return audio_model_part(*find_audio_model(web_config.app_config, name))
    if kind == KIND_ENHANCER:
        repo, revision = _enhancer_model(name, web_config.app_config)
        return enhancer_part(repo, revision, web_config.app_config, platform_key=sys.platform, tier=name)
    if kind == KIND_MATTE:
        _require_matte_row(name)
        return matte_part()
    raise ValueError(f"Unknown download kind '{kind}'.")


def row_download_needed(
    kind: str,
    name: str,
    web_config: WebUiConfig,
    *,
    downloaded: bool | None,
    unavailable: str | None = None,
    present: Callable[[DownloadTarget], bool] = target_present_offline,
) -> bool:
    """Return whether the row still has something to download, judged offline from what is on disk.

    A model built on first use (Ideogram 4 Instant, MOSS-Image) needs its sources; any other image row needs a download when it is not
    *downloaded* (no planner runs, so no file list is read); video, SeedVR2 and enhancer rows need one when a
    target of their plan is not present. A video plan that cannot be made falls back to *downloaded*. Rows of
    unknown status (``downloaded`` is ``None``) and enhancer rows that are *unavailable* need none. The Background
    removal row needs a download when either of its two repos is not present.

    Args:
        kind: ``"model"``, ``"enhancer"`` or ``"matte"``.
        name: The row's name or tier.
        web_config: The Web UI config with its model inventory.
        downloaded: The row's ``downloaded`` status, ``None`` when unknown.
        unavailable: The reason an enhancer tier cannot run here, if any.
        present: Tells whether a target is fully on disk without using the network.

    Raises:
        ValueError: If the row does not exist or the config cannot be planned from.
    """
    if kind == KIND_ENHANCER:
        repo, revision = _enhancer_model(name, web_config.app_config)
        return enhancer_download_needed(repo, revision, unavailable=unavailable, present=present)
    if kind == KIND_MATTE:
        _require_matte_row(name)
        return matte_download_needed(present=present)
    image = _image_entry(name, web_config)
    if image is not None:
        built_plan = built_model_plan(image.resolved_path, config=web_config.app_config, platform_key=sys.platform)
        if built_plan is not None:
            return _any_missing(built_plan, present)
        return downloaded is False
    if downloaded is True and _is_seedvr2_row(name, web_config):
        return False
    targets = _model_row_plan(name, web_config).targets
    if targets is None:
        return downloaded is False
    return _any_missing(targets, present)


def enhancer_download_needed(
    repo: str,
    revision: str | None,
    *,
    unavailable: str | None,
    present: Callable[[DownloadTarget], bool] = target_present_offline,
) -> bool:
    """Return whether the enhancer model *repo* still has files to download; never for a tier that cannot run here."""
    if unavailable:
        return False
    return _any_missing(enhancer_plan(repo, revision), present)


def matte_download_needed(*, present: Callable[[DownloadTarget], bool] = target_present_offline) -> bool:
    """Return whether the background removal models still have files to download; a plan that cannot be made needs none."""
    return _any_missing(matte_plan(), present)


def parts_summary(
    parts: Sequence[JobDownloadPart],
    *,
    check: Callable[[DownloadTarget], TargetCheck | None] = check_target,
    present: Callable[[DownloadTarget], bool] = target_present_offline,
    space: Callable[..., SpaceNeed | None] = space_need,
    cache_dir: Callable[[], Path] = huggingface_cache_dir,
    licences: Sequence[LicenceRecord] = (),
) -> dict[str, Any]:
    """Describe what the downloads of *parts* (a job's, or a Models-page row's one part) would fetch, for the download dialog.

    A part is listed when it has files to fetch, or with ``bytes`` ``None`` (and the result ``unchecked``) when it
    cannot be planned or the Hub cannot be reached. Targets already on disk are skipped without a Hub call.

    Raises:
        RuntimeError: If the Hub refused a request (gated, missing repo or revision, server error); the message names the repo.
    """
    listed: list[dict[str, Any]] = []
    checked: list[tuple[DownloadTarget, TargetCheck]] = []
    unchecked = False
    for part in parts:
        entry, part_unchecked, part_checked = _part_entry(part, check=check, present=present)
        checked.extend(part_checked)
        unchecked = unchecked or part_unchecked
        if entry is not None:
            listed.append(entry)
    sizes = [entry["bytes"] for entry in listed]
    need = space(checked, cache_dir=cache_dir()) if checked else None
    return {
        "parts": listed,
        "total_bytes": None if any(size is None for size in sizes) else sum(sizes),
        "space": None if need is None else {"path": str(need.path), "needed_bytes": need.needed_bytes, "free_bytes": need.free_bytes, "enough": need.enough},
        "unchecked": unchecked,
        "licences": [{"model": entry["model"], **record} for entry in listed for record in licence_payload(licences, entry["model"])],
    }


def _part_entry(
    part: JobDownloadPart,
    *,
    check: Callable[[DownloadTarget], TargetCheck | None],
    present: Callable[[DownloadTarget], bool],
) -> tuple[dict[str, Any] | None, bool, list[tuple[DownloadTarget, TargetCheck]]]:
    """Return a part's dialog entry (``None`` when it has nothing to fetch), whether it is unchecked and its checked targets."""
    if part.targets is None:
        return _entry(part, None), True, []
    missing: list[tuple[DownloadTarget, TargetCheck]] = []
    for target in part.targets:
        if present(target):
            continue
        result = check(target)
        if result is None:
            return _entry(part, None), True, missing
        if result.missing_files > 0:
            missing.append((target, result))
    if not missing:
        return None, False, []
    sizes = [result.missing_bytes for _, result in missing]
    size = None if any(item is None for item in sizes) else sum(item or 0 for item in sizes)
    return _entry(part, size), False, missing


def _entry(part: JobDownloadPart, size: int | None) -> dict[str, Any]:
    """Return one part's row in the download plan: its label, its model (the key of its licences) and what it fetches."""
    return {"label": part.label, "model": part.model_id, "bytes": size}


def _model_row_plan(name: str, web_config: WebUiConfig) -> JobDownloadPart:
    """Return the plan of an image, SeedVR2 or video row; ``ValueError`` when there is no such row."""
    app_config = web_config.app_config
    image = _image_entry(name, web_config)
    if image is not None:
        return image_model_part(name, image.resolved_path, family=image.family, config=app_config, platform_key=sys.platform)
    setup = web_config.seedvr2
    if _is_seedvr2_row(name, web_config) and setup is not None:
        return seedvr2_tier_part(setup.tier.model, setup.weights, Path(web_config.models_dir))
    video = next((entry for entry in web_config.video_inventory if entry.name == name), None)
    if video is not None:
        return video_model_part(video.name, video.resolved_path, family=video.family, config=app_config, upscale=True)
    raise ValueError(f"No model named '{name}' to download.")


def _is_seedvr2_row(name: str, web_config: WebUiConfig) -> bool:
    """Return whether *name* is the Models-page row of the SeedVR2 tier this machine uses."""
    setup = web_config.seedvr2
    return setup is not None and name == seedvr2_row_name(setup.tier.model, setup.weights)


def _image_entry(name: str, web_config: WebUiConfig) -> ImageInventoryEntry | None:
    """Return the image inventory entry called *name*, if any."""
    return next((entry for entry in web_config.image_inventory if entry.name == name), None)


def _enhancer_model(tier: str, app_config: dict[str, Any]) -> tuple[str, str | None]:
    """Return the ``(repo, revision)`` configured for the enhancer *tier*; ``ValueError`` when the tier has none."""
    model = enhancer_tier_models(app_config, platform_key=sys.platform).get(tier)
    if model is None:
        raise ValueError(f"No enhancer model for '{tier}' to download.")
    return model


def _require_matte_row(name: str) -> None:
    """Raise ``ValueError`` unless *name* is the one Background removal row."""
    if name != MATTE_ROW_NAME:
        raise ValueError(f"No background removal row '{name}' to download.")


def _any_missing(targets: Sequence[DownloadTarget] | None, present: Callable[[DownloadTarget], bool]) -> bool:
    """Return whether any of *targets* is not fully on disk (``None`` or no targets: nothing to fetch)."""
    return any(not present(target) for target in targets or ())
