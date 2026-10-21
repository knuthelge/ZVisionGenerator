"""Plan the downloads behind a Models-page row: the targets, whether the row needs any, and what they would fetch."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.downloads.check import check_target
from zvisiongenerator.downloads.plans import enhancer_plan, image_plan, instant_plan, seedvr2_plan, video_plan
from zvisiongenerator.downloads.presence import target_present_offline
from zvisiongenerator.downloads.space import space_need
from zvisiongenerator.utils.config import enhancer_tier_models
from zvisiongenerator.utils.ideogram4_instant import is_instant_dir
from zvisiongenerator.utils.model_files import huggingface_cache_dir
from zvisiongenerator.utils.seedvr2 import seedvr2_row_name

if TYPE_CHECKING:
    from zvisiongenerator.downloads.space import SpaceNeed
    from zvisiongenerator.downloads.target import DownloadTarget, TargetCheck
    from zvisiongenerator.web.config import WebUiConfig
    from zvisiongenerator.web.model_inventory import ImageInventoryEntry

__all__ = [
    "KIND_ENHANCER",
    "KIND_MODEL",
    "PlanUnchecked",
    "RowPlan",
    "enhancer_download_needed",
    "plan_summary",
    "row_download_needed",
    "row_plan",
]

KIND_MODEL = "model"
KIND_ENHANCER = "enhancer"


class PlanUnchecked(Exception):  # noqa: N818 - a state, not an error class; callers map it to HTTP 503
    """Raised when a plan's size cannot be determined because the Hugging Face Hub cannot be reached."""


@dataclass(frozen=True)
class RowPlan:
    """The downloads a Models-page row needs.

    Attributes:
        kind: ``"model"`` (image, video and SeedVR2 rows, by name) or ``"enhancer"`` (by tier).
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

    An Ideogram 4 Instant folder needs its sources; any other image row needs a download when it is not
    *downloaded* (no planner runs, so no file list is read); video, SeedVR2 and enhancer rows need one when a
    target of their plan is not present. A video plan that cannot be made falls back to *downloaded*. Rows of
    unknown status (``downloaded`` is ``None``) and enhancer rows that are *unavailable* need none.

    Args:
        kind: ``"model"`` or ``"enhancer"``.
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
    image = _image_entry(name, web_config)
    if image is not None:
        if is_instant_dir(image.resolved_path):
            return _any_missing(instant_plan(image.resolved_path, config=web_config.app_config, platform_key=sys.platform), present)
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


def plan_summary(
    plan: RowPlan,
    *,
    check: Callable[[DownloadTarget], TargetCheck | None] = check_target,
    present: Callable[[DownloadTarget], bool] = target_present_offline,
    space: Callable[..., SpaceNeed | None] = space_need,
    cache_dir: Callable[[], Path] = huggingface_cache_dir,
) -> dict[str, Any]:
    """Describe what downloading *plan* would fetch: parts with their file counts and sizes, the total and the free space.

    Targets already on disk are skipped; each other one is dry-run against the Hub. ``parts`` is empty and ``space``
    is ``None`` when nothing is missing; ``total_bytes`` is ``None`` when any part's size is unknown. ``space`` is
    the main download drive's estimate only.

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
    }


def _model_row_plan(name: str, web_config: WebUiConfig) -> RowPlan:
    """Return the plan of an image, SeedVR2 or video row; ``ValueError`` when there is no such row."""
    app_config = web_config.app_config
    image = _image_entry(name, web_config)
    if image is not None:
        return RowPlan(KIND_MODEL, name, name, image_plan(image.resolved_path, family=image.family, config=app_config, platform_key=sys.platform))
    setup = web_config.seedvr2
    if _is_seedvr2_row(name, web_config) and setup is not None:
        return RowPlan(KIND_MODEL, name, name, seedvr2_plan(setup.tier.model, models_dir=Path(web_config.models_dir), weights=setup.weights))
    video = next((entry for entry in web_config.video_inventory if entry.name == name), None)
    if video is not None:
        return RowPlan(KIND_MODEL, name, name, video_plan(video.resolved_path, family=video.family, config=app_config, model_name=video.name, upscale=True))
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


def _enhancer_label(tier: str) -> str:
    """Return how an enhancer tier reads in messages, such as "Smart enhancer model"."""
    return f"{tier.capitalize()} enhancer model"


def _any_missing(targets: Sequence[DownloadTarget] | None, present: Callable[[DownloadTarget], bool]) -> bool:
    """Return whether any of *targets* is not fully on disk (``None`` or no targets: nothing to fetch)."""
    return any(not present(target) for target in targets or ())
