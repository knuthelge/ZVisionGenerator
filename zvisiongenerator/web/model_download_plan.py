"""Plan the downloads behind a Models-page row: the targets, whether the row needs any, and what they would fetch."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.downloads.check import check_target
from zvisiongenerator.downloads.plans import built_model_plan, enhancer_plan, image_plan, matte_plan, seedvr2_plan, video_plan
from zvisiongenerator.downloads.presence import target_present_offline
from zvisiongenerator.downloads.space import space_need
from zvisiongenerator.utils.config import enhancer_tier_models, model_display_name
from zvisiongenerator.utils.licences import MATTE_LICENCE_MODEL, licence_payload
from zvisiongenerator.utils.matte_models import MATTE_LABEL, MATTE_ROW_NAME
from zvisiongenerator.utils.model_files import huggingface_cache_dir
from zvisiongenerator.utils.seedvr2 import seedvr2_row_name

if TYPE_CHECKING:
    from zvisiongenerator.downloads.space import SpaceNeed
    from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck
    from zvisiongenerator.web.config import WebUiConfig
    from zvisiongenerator.web.job_downloads import JobDownloadPart
    from zvisiongenerator.web.model_inventory import ImageInventoryEntry

__all__ = [
    "KIND_ENHANCER",
    "KIND_MATTE",
    "KIND_MODEL",
    "PlanUnchecked",
    "RowPlan",
    "enhancer_download_needed",
    "matte_download_needed",
    "parts_summary",
    "plan_summary",
    "row_download_needed",
    "row_plan",
]

KIND_MODEL = "model"
KIND_ENHANCER = "enhancer"
KIND_MATTE = "matte"


class PlanUnchecked(Exception):  # noqa: N818 - a state, not an error class; callers map it to HTTP 503
    """Raised when a plan's size cannot be determined because the Hugging Face Hub cannot be reached."""


@dataclass(frozen=True)
class RowPlan:
    """The downloads a Models-page row needs.

    Attributes:
        kind: ``"model"`` (image, video and SeedVR2 rows, by name), ``"enhancer"`` (by tier) or ``"matte"`` (the one Background removal row).
        name: The row's name or tier.
        label: How the row reads in messages.
        targets: The targets to fetch, or ``None`` when the plan cannot be made now (no connection to read the model's file list).
    """

    kind: str
    name: str
    label: str
    targets: tuple[DownloadTarget, ...] | None


def row_plan(kind: str, name: str, web_config: WebUiConfig) -> RowPlan:
    """Return the download plan of the Models-page row *name* of *kind*.

    Raises:
        ValueError: If *kind* is unknown or no such row exists.
    """
    if kind == KIND_MODEL:
        return _model_row_plan(name, web_config)
    if kind == KIND_ENHANCER:
        repo, revision = _enhancer_model(name, web_config.app_config)
        return RowPlan(kind, name, _enhancer_label(name), enhancer_plan(repo, revision))
    if kind == KIND_MATTE:
        _require_matte_row(name)
        return RowPlan(kind, name, MATTE_LABEL, matte_plan())
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


def plan_summary(
    plan: RowPlan,
    *,
    check: Callable[[DownloadTarget], TargetCheck | None] = check_target,
    present: Callable[[DownloadTarget], bool] = target_present_offline,
    space: Callable[..., SpaceNeed | None] = space_need,
    cache_dir: Callable[[], Path] = huggingface_cache_dir,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe what downloading *plan* would fetch: parts with their file counts and sizes, the total, the free space and the licences.

    Targets already on disk are skipped; each other one is dry-run against the Hub. ``parts`` is empty and ``space``
    is ``None`` when nothing is missing; ``total_bytes`` is ``None`` when any part's size is unknown. ``space`` is
    the main download drive's estimate only. ``licences`` holds the records of the row's model (the background removal
    row's own) for the dialog, each with the row's ``name`` as its ``model``; it is empty without a *config*.

    Raises:
        PlanUnchecked: If the plan cannot be made now or a target cannot be checked because the Hub is unreachable.
        RuntimeError: If the Hub refused a request (gated, missing repo or revision, server error); the message names the repo.
    """
    if plan.targets is None:
        raise PlanUnchecked(f"Could not reach Hugging Face to check the download of {plan.label}.")
    checked: list[tuple[DownloadTarget, TargetCheck]] = []
    for target in plan.targets:
        if present(target):
            continue
        result = check(target)
        if result is None:
            raise PlanUnchecked(f"Could not reach Hugging Face to check the download of {plan.label}.")
        if result.missing_files > 0:
            checked.append((target, result))
    sizes = [result.missing_bytes for _, result in checked]
    need = space(checked, cache_dir=cache_dir()) if checked else None
    return {
        "kind": plan.kind,
        "name": plan.name,
        "label": plan.label,
        "parts": [{"repo": target.repo_id, "files": result.missing_files, "bytes": result.missing_bytes} for target, result in checked],
        "total_bytes": None if any(size is None for size in sizes) else sum(size or 0 for size in sizes),
        "space": None if need is None else {"path": str(need.path), "needed_bytes": need.needed_bytes, "free_bytes": need.free_bytes, "enough": need.enough},
        "licences": _row_licences(plan, config),
    }


def parts_summary(
    parts: Sequence[JobDownloadPart],
    *,
    check: Callable[[DownloadTarget], TargetCheck | None] = check_target,
    present: Callable[[DownloadTarget], bool] = target_present_offline,
    space: Callable[..., SpaceNeed | None] = space_need,
    cache_dir: Callable[[], Path] = huggingface_cache_dir,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe what the downloads of a job's *parts* would fetch, for the download dialog.

    A part is listed when it has files to fetch or when it cannot be sized: its targets cannot be planned now, or the
    Hub cannot be reached to dry-run one of them. Such a part has ``files`` 0 and ``bytes`` ``None``, and the result
    is ``unchecked``. A target already on disk is skipped without a Hub call, so nothing missing means no call and no
    parts. A listed part's ``repo`` is its first target that has files to fetch (or, unchecked, its first target).

    Returns:
        ``parts`` (``label``, ``model``, ``repo``, ``files``, ``bytes`` each, in the order given), ``total_bytes``
        (``None`` when any size is unknown), ``space`` (the main drive's estimate of the checked part, or ``None``),
        ``unchecked`` and ``licences`` (the records of the listed parts' models, each with the ``model`` it covers; empty without a *config*).

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
        "licences": [{"model": entry["model"], **record} for entry in listed for record in _licences(config, entry["model"])],
    }


def _licences(config: Mapping[str, Any] | None, model_id: str) -> list[dict[str, Any]]:
    """Return the licence records of *model_id*; none without a *config*."""
    return [] if config is None else licence_payload(config, model_id)


def _row_licences(plan: RowPlan, config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Return the licence records of a Models-page row, each tagged with the row's ``name``; only image, video and background removal rows have any."""
    if plan.kind == KIND_ENHANCER:
        return []
    model_id = MATTE_LICENCE_MODEL if plan.kind == KIND_MATTE else plan.name
    return [{"model": plan.name, **record} for record in _licences(config, model_id)]


def _part_entry(
    part: JobDownloadPart,
    *,
    check: Callable[[DownloadTarget], TargetCheck | None],
    present: Callable[[DownloadTarget], bool],
) -> tuple[dict[str, Any] | None, bool, list[tuple[DownloadTarget, TargetCheck]]]:
    """Return a part's dialog entry (``None`` when it has nothing to fetch), whether it is unchecked and its checked targets."""
    if part.targets is None:
        return _entry(part, repo="", files=0, size=None), True, []
    missing: list[tuple[DownloadTarget, TargetCheck]] = []
    for target in part.targets:
        if present(target):
            continue
        result = check(target)
        if result is None:
            return _entry(part, repo=target.repo_id, files=0, size=None), True, missing
        if result.missing_files > 0:
            missing.append((target, result))
    if not missing:
        return None, False, []
    sizes = [result.missing_bytes for _, result in missing]
    size = None if any(item is None for item in sizes) else sum(item or 0 for item in sizes)
    return _entry(part, repo=missing[0][0].repo_id, files=sum(result.missing_files for _, result in missing), size=size), False, missing


def _entry(part: JobDownloadPart, *, repo: str, files: int, size: int | None) -> dict[str, Any]:
    """Return one part's row in the job download plan."""
    return {"label": part.label, "model": part.model_id, "repo": repo, "files": files, "bytes": size}


def _model_row_plan(name: str, web_config: WebUiConfig) -> RowPlan:
    """Return the plan of an image, SeedVR2 or video row; ``ValueError`` when there is no such row."""
    app_config = web_config.app_config
    image = _image_entry(name, web_config)
    if image is not None:
        return RowPlan(KIND_MODEL, name, model_display_name(app_config, name), image_plan(image.resolved_path, family=image.family, config=app_config, platform_key=sys.platform))
    setup = web_config.seedvr2
    if _is_seedvr2_row(name, web_config) and setup is not None:
        return RowPlan(KIND_MODEL, name, name, seedvr2_plan(setup.tier.model, models_dir=Path(web_config.models_dir), weights=setup.weights))
    video = next((entry for entry in web_config.video_inventory if entry.name == name), None)
    if video is not None:
        return RowPlan(KIND_MODEL, name, model_display_name(app_config, name), video_plan(video.resolved_path, family=video.family, config=app_config, model_name=video.name, upscale=True))
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


def _enhancer_label(tier: str) -> str:
    """Return how an enhancer tier reads in messages, such as "Smart enhancer model"."""
    return f"{tier.capitalize()} enhancer model"


def _any_missing(targets: Sequence[DownloadTarget] | None, present: Callable[[DownloadTarget], bool]) -> bool:
    """Return whether any of *targets* is not fully on disk (``None`` or no targets: nothing to fetch)."""
    return any(not present(target) for target in targets or ())
