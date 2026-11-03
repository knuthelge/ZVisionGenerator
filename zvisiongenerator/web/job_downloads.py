"""Plan the downloads of one Web UI job: the parts the runner fetches and the plan routes show, from one place."""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from zvisiongenerator.backends import get_seedvr2_upscaler
from zvisiongenerator.backends import prompt_enhancer_session
from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.video_types import LtxVideoUpscaleRequest, VideoFileUpscaleRequest, VideoGenerationRequest
from zvisiongenerator.downloads import DownloadTarget, audio_plan, enhancer_plan, image_plan, matte_plan, seedvr2_plan, video_plan
from zvisiongenerator.utils.audio_preset import AudioPreset
from zvisiongenerator.utils.config import enhancer_tier_models, model_display_name, model_reference, resolve_enhancer_model
from zvisiongenerator.utils.licences import MATTE_LICENCE_MODEL
from zvisiongenerator.utils.matte_models import MATTE_LABEL
from zvisiongenerator.utils.paths import get_ziv_data_dir
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings
from zvisiongenerator.utils.provenance import UPSCALE_METHOD_SEEDVR2
from zvisiongenerator.utils.seedvr2 import SeedVR2Weights, seedvr2_tier_label

__all__ = [
    "ROLE_ENHANCER",
    "ROLE_MATTE",
    "ROLE_MODEL",
    "JobDownloadPart",
    "audio_model_part",
    "enhancer_part",
    "image_model_part",
    "job_download_parts",
    "ltx_video_upscale_part",
    "drop_in_flight",
    "matte_part",
    "seedvr2_part",
    "seedvr2_tier_part",
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


def image_model_part(name: str, model_ref: str, *, family: str, config: dict[str, Any], platform_key: str) -> JobDownloadPart:
    """Return the part for the image model *name* (an alias or a path) that loads from *model_ref*, labelled as the Web UI names it."""
    label = model_display_name(config, name)
    return JobDownloadPart(label, label, name, ROLE_MODEL, image_plan(model_ref, family=family, config=config, platform_key=platform_key))


def video_model_part(name: str, model_ref: str, *, family: str, config: dict[str, Any], upscale: bool) -> JobDownloadPart:
    """Return the part for the video model *name* that loads from *model_ref* (with the upscale's files when it *upscale*s).

    Raises:
        ValueError: If the ``video_model_presets`` config entries of the family are malformed.
    """
    label = model_display_name(config, name)
    return JobDownloadPart(label, label, name, ROLE_MODEL, video_plan(model_ref, family=family, config=config, model_name=name, upscale=upscale))


def ltx_video_upscale_part(request: LtxVideoUpscaleRequest, config: dict[str, Any]) -> JobDownloadPart:
    """Return the part for the LTX model that refines an upscale of a video file, with the upscale's files.

    Raises:
        ValueError: If the ``video_model_presets`` config entries of the family are malformed.
    """
    return video_model_part(request.model_name, request.model_ref, family="ltx", config=config, upscale=True)


def audio_model_part(preset: AudioPreset, platform_key: str) -> JobDownloadPart:
    """Return the part for the audio model *preset* an audio job generates with on the platform *platform_key*."""
    return JobDownloadPart(preset.label, preset.id, preset.id, ROLE_MODEL, audio_plan(preset, platform_key))


def matte_part() -> JobDownloadPart:
    """Return the part for the background removal models of a transparent job."""
    return JobDownloadPart(MATTE_LABEL, MATTE_LABEL, MATTE_LICENCE_MODEL, ROLE_MATTE, matte_plan())


def seedvr2_tier_part(model_name: str, weights: SeedVR2Weights, models_dir: Path) -> JobDownloadPart:
    """Return the part for SeedVR2 tier *model_name* as *weights* run it, with its files under *models_dir*."""
    label = seedvr2_tier_label(model_name, weights)
    return JobDownloadPart(label, label, model_name, ROLE_MODEL, seedvr2_plan(model_name, models_dir=models_dir, weights=weights))


def seedvr2_part(model_name: str | None) -> JobDownloadPart:
    """Return the part for SeedVR2 tier *model_name*; its targets are ``None`` when this platform has no SeedVR2 engine."""
    upscaler = get_seedvr2_upscaler()
    if upscaler is None:
        # The load reports that this platform has no SeedVR2 engine.
        return JobDownloadPart(str(model_name), str(model_name), str(model_name), ROLE_MODEL, None)
    return seedvr2_tier_part(str(model_name), upscaler.weights, get_ziv_data_dir() / "models")


def upscale_model_part(request: ImageGenerationRequest, model_ref: str, config: dict[str, Any], *, platform_key: str) -> JobDownloadPart:
    """Return the part for the model an image upscale loads: the SeedVR2 tier or the creative refinement model."""
    if request.upscale_method == UPSCALE_METHOD_SEEDVR2:
        return seedvr2_part(request.upscale_model)
    return image_model_part(request.model_name or model_ref, model_ref, family=request.model_family, config=config, platform_key=platform_key)


def enhancer_part(repo: str, revision: str | None, config: dict[str, Any], *, platform_key: str, tier: str | None = None) -> JobDownloadPart:
    """Return the part for the prompt enhancer ``(repo, revision)`` (no targets for a local folder).

    The label is the tier's ("Smart enhancer model"): *tier* (``fast``, ``smart`` or ``custom``) when given, else the
    configured tier whose model this is. A model no tier names is labelled by its ``repo[@revision]``.
    """
    return _enhancer_part(repo, revision, tier or _enhancer_tier(config, platform_key, repo, revision), enhancer_plan(repo, revision))


def job_download_parts(
    request: ImageGenerationRequest | VideoGenerationRequest | VideoFileUpscaleRequest | LtxVideoUpscaleRequest,
    config: dict[str, Any],
    *,
    platform_key: str,
    model_ref: str | None = None,
    args: argparse.Namespace | None = None,
    enhance_by_set: dict[str, list[EnhanceSettings | None]] | None = None,
) -> tuple[JobDownloadPart, ...]:
    """Return every part the job for *request* downloads before it loads, in the order the job downloads them.

    *args* and *enhance_by_set* decide the enhancer part; *model_ref* is required except for a video-file upscale. An enhancer
    that cannot be planned offline is a part with ``targets=None``. An audio job has one part, :func:`audio_model_part`.

    Raises:
        ValueError: If *model_ref* is missing for a job that needs it, no enhancer is configured, or a video preset is malformed.
    """
    if isinstance(request, VideoFileUpscaleRequest):
        return (seedvr2_part(request.upscale_model),)
    if isinstance(request, LtxVideoUpscaleRequest):
        return (ltx_video_upscale_part(request, config),)
    if model_ref is None:
        raise ValueError("model_ref is required to plan an image or video job.")
    if isinstance(request, VideoGenerationRequest):
        return (
            *_enhancer_parts(args, enhance_by_set, config, platform_key),
            video_model_part(request.model_name or model_ref, model_ref, family=request.model_family, config=config, upscale=bool(request.upscale)),
        )
    matte = (matte_part(),) if request.transparent else ()
    if request.upscale_source is not None:
        return (upscale_model_part(request, model_ref, config, platform_key=platform_key), *matte)
    image = image_model_part(request.model_name or model_ref, model_ref, family=request.model_family, config=config, platform_key=platform_key)
    return (*_enhancer_parts(args, enhance_by_set, config, platform_key), image, *matte)


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


def _enhancer_parts(args: argparse.Namespace | None, enhance_by_set: dict[str, list[EnhanceSettings | None]] | None, config: dict[str, Any], platform_key: str) -> tuple[JobDownloadPart, ...]:
    """Return the enhancer part of a job that enhances (the inputs of ``run_preflight``), else nothing; unchecked when offline."""
    cli_model = getattr(args, "enhance_model", None)
    budget_bytes = prompt_enhancer_session.smart_budget_bytes(cli_model)
    try:
        reference = prompt_enhancer_session.plan_job_enhancer(
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
    return (enhancer_part(*reference, config, platform_key=platform_key),)


def _unchecked_enhancer(config: dict[str, Any], platform_key: str, cli_model: str | None, budget_bytes: int | None) -> JobDownloadPart:
    """Return the enhancer part named by the configuration with no targets, for when the model cannot be checked."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the Smart-unavailable warning was already issued by the planning attempt
        repo, revision = resolve_enhancer_model(config, platform_key=platform_key, cli_model=cli_model, budget_bytes=budget_bytes)
    return _enhancer_part(repo, revision, _enhancer_tier(config, platform_key, repo, revision), None)


def _enhancer_part(repo: str, revision: str | None, tier: str | None, targets: Targets) -> JobDownloadPart:
    """Return an enhancer part labelled "Smart enhancer model" for its *tier*, or by its reference when it has none."""
    reference = model_reference(repo, revision)
    label = f"{tier.capitalize()} enhancer model" if tier else reference
    return JobDownloadPart(label, label, reference, ROLE_ENHANCER, targets)


def _enhancer_tier(config: dict[str, Any], platform_key: str, repo: str, revision: str | None) -> str | None:
    """Return the configured enhancer tier whose model is ``(repo, revision)``, or ``None`` for a model no tier names."""
    return next((tier for tier, model in enhancer_tier_models(config, platform_key=platform_key).items() if model == (repo, revision)), None)
