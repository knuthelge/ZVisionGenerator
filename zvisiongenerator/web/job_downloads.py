"""Plan the downloads of one Web UI job: the parts a job fetches before it loads, with their event names and targets.

A part is one model the job needs (the generation model, the prompt enhancer, the background removal models, a SeedVR2
tier). The runner downloads each part with the targets and ``event_model`` given here, and the plan routes show the same
parts to the user, so what a job downloads and what the user is told it downloads come from one place.

The builders are pure: they read their inputs and the planners passed in and download nothing. The planners default to
the real ones; the runner passes the ones it binds so a test can replace them in one place.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from zvisiongenerator.backends import get_seedvr2_upscaler
from zvisiongenerator.backends import prompt_enhancer_session
from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.video_types import LtxVideoUpscaleRequest, VideoFileUpscaleRequest, VideoGenerationRequest
from zvisiongenerator.downloads import DownloadTarget, enhancer_plan, image_plan, matte_plan, seedvr2_plan, video_plan
from zvisiongenerator.utils.config import model_reference, resolve_enhancer_model
from zvisiongenerator.utils.licences import MATTE_LICENCE_MODEL
from zvisiongenerator.utils.matte_models import MATTE_LABEL
from zvisiongenerator.utils.paths import get_ziv_data_dir
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings
from zvisiongenerator.utils.provenance import UPSCALE_METHOD_SEEDVR2
from zvisiongenerator.utils.seedvr2 import seedvr2_tier_label

__all__ = [
    "ROLE_ENHANCER",
    "ROLE_MATTE",
    "ROLE_MODEL",
    "DownloadPlanners",
    "JobDownloadPart",
    "enhancer_part",
    "image_model_part",
    "job_download_parts",
    "ltx_video_upscale_part",
    "drop_in_flight",
    "matte_part",
    "seedvr2_part",
    "upscale_model_part",
    "video_model_part",
]

ROLE_MODEL = "model"
"""The generation or upscale model (the ``role`` of its download events)."""

ROLE_ENHANCER = "enhancer"
"""The prompt enhancer."""

ROLE_MATTE = "matte"
"""The background removal models of a transparent job."""

type Targets = tuple[DownloadTarget, ...] | None


@dataclass(frozen=True)
class JobDownloadPart:
    """One model a job downloads before it loads.

    Attributes:
        label: The name the download dialog shows.
        event_model: The ``model`` value the job's ``model_downloading`` events carry.
        model_id: The model's identifier for looking up its licence records.
        role: ``model``, ``enhancer`` or ``matte``; the ``role`` of the job's download events.
        targets: The files to fetch (``()`` when nothing is to be fetched), or ``None`` when they cannot be planned
            now (offline); a part with ``None`` is *unchecked* and a job loads without downloading it.
    """

    label: str
    event_model: str
    model_id: str
    role: str
    targets: Targets


@dataclass(frozen=True)
class DownloadPlanners:
    """The functions that turn a model into download targets, passed in so the builders stay pure and testable."""

    image: Callable[..., Targets] = image_plan
    video: Callable[..., Targets] = video_plan
    matte: Callable[[], Targets] = matte_plan
    seedvr2: Callable[..., tuple[DownloadTarget, ...]] = seedvr2_plan
    enhancer: Callable[[str, str | None], tuple[DownloadTarget, ...]] = enhancer_plan
    plan_enhancer: Callable[..., tuple[str, str | None] | None] = prompt_enhancer_session.plan_job_enhancer
    seedvr2_upscaler: Callable[[], Any] = get_seedvr2_upscaler
    data_dir: Callable[[], Path] = get_ziv_data_dir


def image_model_part(request: ImageGenerationRequest, model_ref: str, config: dict[str, Any], *, platform_key: str, planners: DownloadPlanners | None = None) -> JobDownloadPart:
    """Return the part for the image model an image job generates with."""
    planners = planners or DownloadPlanners()
    name = request.model_name or model_ref
    return JobDownloadPart(name, name, name, ROLE_MODEL, planners.image(model_ref, family=request.model_family, config=config, platform_key=platform_key))


def video_model_part(request: VideoGenerationRequest, model_ref: str, config: dict[str, Any], *, planners: DownloadPlanners | None = None) -> JobDownloadPart:
    """Return the part for the video model a video job generates with (with the upscale's files when it upscales).

    Raises:
        ValueError: If the ``video_model_presets`` config entries of the family are malformed.
    """
    planners = planners or DownloadPlanners()
    name = request.model_name or model_ref
    targets = planners.video(model_ref, family=request.model_family, config=config, model_name=name, upscale=bool(request.upscale))
    return JobDownloadPart(name, name, name, ROLE_MODEL, targets)


def ltx_video_upscale_part(request: LtxVideoUpscaleRequest, config: dict[str, Any], *, planners: DownloadPlanners | None = None) -> JobDownloadPart:
    """Return the part for the LTX model that refines an upscale of a video file, with the upscale's files.

    Raises:
        ValueError: If the ``video_model_presets`` config entries of the family are malformed.
    """
    planners = planners or DownloadPlanners()
    name = request.model_name
    targets = planners.video(request.model_ref, family="ltx", config=config, model_name=name, upscale=True)
    return JobDownloadPart(name, name, name, ROLE_MODEL, targets)


def matte_part(*, planners: DownloadPlanners | None = None) -> JobDownloadPart:
    """Return the part for the background removal models of a transparent job."""
    planners = planners or DownloadPlanners()
    return JobDownloadPart(MATTE_LABEL, MATTE_LABEL, MATTE_LICENCE_MODEL, ROLE_MATTE, planners.matte())


def seedvr2_part(model_name: str | None, *, planners: DownloadPlanners | None = None) -> JobDownloadPart:
    """Return the part for SeedVR2 tier *model_name*; its targets are ``None`` when this platform has no SeedVR2 engine."""
    planners = planners or DownloadPlanners()
    upscaler = planners.seedvr2_upscaler()
    if upscaler is None:
        # The load reports that this platform has no SeedVR2 engine.
        return JobDownloadPart(str(model_name), str(model_name), str(model_name), ROLE_MODEL, None)
    label = seedvr2_tier_label(model_name, upscaler.weights)
    targets = planners.seedvr2(model_name, models_dir=planners.data_dir() / "models", weights=upscaler.weights)
    return JobDownloadPart(label, label, str(model_name), ROLE_MODEL, targets)


def upscale_model_part(request: ImageGenerationRequest, model_ref: str, config: dict[str, Any], *, platform_key: str, planners: DownloadPlanners | None = None) -> JobDownloadPart:
    """Return the part for the model an image upscale loads: the SeedVR2 tier or the creative refinement model."""
    if request.upscale_method == UPSCALE_METHOD_SEEDVR2:
        return seedvr2_part(request.upscale_model, planners=planners)
    return image_model_part(request, model_ref, config, platform_key=platform_key, planners=planners)


def enhancer_part(repo: str, revision: str | None, *, planners: DownloadPlanners | None = None) -> JobDownloadPart:
    """Return the part for the prompt enhancer ``(repo, revision)`` (no targets for a local folder)."""
    planners = planners or DownloadPlanners()
    return JobDownloadPart(model_reference(repo, revision), repo, model_reference(repo, revision), ROLE_ENHANCER, planners.enhancer(repo, revision))


def job_download_parts(
    request: ImageGenerationRequest | VideoGenerationRequest | VideoFileUpscaleRequest | LtxVideoUpscaleRequest,
    args: argparse.Namespace | None,
    prompts_data: Mapping[str, list[tuple[str, str | None]]] | None,
    enhance_by_set: dict[str, list[EnhanceSettings | None]] | None,
    config: dict[str, Any],
    *,
    platform_key: str,
    model_ref: str | None = None,
    planners: DownloadPlanners | None = None,
) -> tuple[JobDownloadPart, ...]:
    """Return every part the job for *request* downloads before it loads, in the order the job downloads them.

    Image Generate: the enhancer when any prompt is enhanced (preflight fetches it first), the image model, and the
    background removal models when transparent. Video: the enhancer, then the video model (with the upscale's files when it upscales). Image upscale:
    the SeedVR2 tier or the creative model, and the background removal models when the source is transparent. Video-file
    upscale: the SeedVR2 tier, or on macOS the LTX model that refines the clip. An enhancer that cannot be planned because Hugging Face is offline is a part with
    ``targets=None``; nothing is downloaded or loaded here.

    Args:
        request: The job's request; its type selects the job.
        args: The job's arguments, read for the enhancer (``no_enhance``, ``enhance``, ``enhance_model``); unused by upscales.
        prompts_data: The job's prompts; unused today, taken so every job type shares one signature.
        enhance_by_set: Per-entry prompt-file ``enhance:`` settings aligned with *prompts_data*.
        config: The loaded config.
        platform_key: Platform key such as ``darwin`` or ``win32``.
        model_ref: The resolved model path or id; required for every job except a video-file upscale.
        planners: The planner functions (defaults to the real ones).

    Raises:
        ValueError: If *model_ref* is missing for a job that needs it, no enhancer is configured, or a video preset is malformed.
    """
    planners = planners or DownloadPlanners()
    if isinstance(request, VideoFileUpscaleRequest):
        return (seedvr2_part(request.upscale_model, planners=planners),)
    if isinstance(request, LtxVideoUpscaleRequest):
        return (ltx_video_upscale_part(request, config, planners=planners),)
    if model_ref is None:
        raise ValueError("model_ref is required to plan an image or video job.")
    if isinstance(request, VideoGenerationRequest):
        return (*_enhancer_parts(args, enhance_by_set, config, platform_key, planners), video_model_part(request, model_ref, config, planners=planners))
    matte = (matte_part(planners=planners),) if request.transparent else ()
    if request.upscale_source is not None:
        return (upscale_model_part(request, model_ref, config, platform_key=platform_key, planners=planners), *matte)
    return (*_enhancer_parts(args, enhance_by_set, config, platform_key, planners), image_model_part(request, model_ref, config, platform_key=platform_key, planners=planners), *matte)


def drop_in_flight(parts: Sequence[JobDownloadPart], active: Iterable[DownloadTarget]) -> tuple[JobDownloadPart, ...]:
    """Return *parts* without the targets that a running download (one of *active*) already fetches.

    A target counts as in flight when its ``key`` is the key of an active flight, so the job that follows the
    flight needs no second question about it. A part with no target left has nothing to fetch and is dropped;
    a part that is unchecked (``targets=None``) stays, since what it would fetch is unknown.
    """
    running = {target.key for target in active}
    if not running:
        return tuple(parts)
    kept: list[JobDownloadPart] = []
    for part in parts:
        if part.targets is None:
            kept.append(part)
            continue
        remaining = tuple(target for target in part.targets if target.key not in running)
        if remaining:
            kept.append(replace(part, targets=remaining))
    return tuple(kept)


def _enhancer_parts(
    args: argparse.Namespace | None, enhance_by_set: dict[str, list[EnhanceSettings | None]] | None, config: dict[str, Any], platform_key: str, planners: DownloadPlanners
) -> tuple[JobDownloadPart, ...]:
    """Return the enhancer part of a job that enhances (the inputs of ``run_preflight``), else nothing; unchecked when offline."""
    cli_model = getattr(args, "enhance_model", None)
    budget_bytes = prompt_enhancer_session.smart_budget_bytes(cli_model)
    try:
        reference = planners.plan_enhancer(
            config,
            platform_key=platform_key,
            disabled=bool(getattr(args, "no_enhance", False)),
            override=getattr(args, "enhance", None),
            enhance_by_set=enhance_by_set,
            cli_model=cli_model,
            budget_bytes=budget_bytes,
        )
    except RuntimeError:
        # Offline with the enhancer missing: the job fails when it needs it, but a plan only reports it as unchecked.
        return (_unchecked_enhancer(config, platform_key, cli_model, budget_bytes),)
    if reference is None:
        return ()
    return (enhancer_part(*reference, planners=planners),)


def _unchecked_enhancer(config: dict[str, Any], platform_key: str, cli_model: str | None, budget_bytes: int | None) -> JobDownloadPart:
    """Return the enhancer part named by the configuration with no targets, for when the model cannot be checked."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the Smart-unavailable warning was already issued by the planning attempt
        repo, revision = resolve_enhancer_model(config, platform_key=platform_key, cli_model=cli_model, budget_bytes=budget_bytes)
    reference = model_reference(repo, revision)
    return JobDownloadPart(reference, repo, reference, ROLE_ENHANCER, None)
