"""Plan the downloads of one Web UI job: the parts the runner fetches and the plan routes show, from one place."""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zvisiongenerator.backends import get_seedvr2_upscaler
from zvisiongenerator.backends import prompt_enhancer_session
from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.video_types import LtxVideoUpscaleRequest, VideoFileUpscaleRequest, VideoGenerationRequest
from zvisiongenerator.downloads import DownloadTarget, audio_plan, enhancer_plan, image_plan, matte_plan, seedvr2_plan, target_present_offline, video_plan
from zvisiongenerator.utils.audio_preset import AudioPreset
from zvisiongenerator.utils.config import enhancer_tier_models, model_display_name, model_reference, resolve_enhancer_model
from zvisiongenerator.utils.licences import ENHANCER_LICENCE_TIERS, MATTE_LICENCE_MODEL, SEEDVR2_LICENCE_MODEL, enhancer_licence_model
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
    "is_committed",
    "job_download_parts",
    "job_model_part",
    "ltx_video_upscale_part",
    "matte_part",
    "missing_targets",
    "planned_targets",
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
        label: The name the download dialog shows, also the ``model`` value of the job's ``model_downloading`` events.
        model_id: The model's identifier for looking up its licence records: the alias, ``matte``, ``seedvr2`` or ``enhancer-<tier>``.
        role: ``model``, ``enhancer`` or ``matte``; the ``role`` of the job's download events.
        targets: The files to fetch (``()`` when nothing is to be fetched), or ``None`` when they cannot be planned
            now (offline); a part with ``None`` is *unchecked* and a job loads without downloading it.
    """

    label: str
    model_id: str
    role: str
    targets: Targets


def image_model_part(name: str, model_ref: str, *, family: str, config: dict[str, Any], platform_key: str) -> JobDownloadPart:
    """Return the part for the image model *name* (an alias or a path) that loads from *model_ref*, labelled as the Web UI names it."""
    label = model_display_name(config, name)
    return JobDownloadPart(label, name, ROLE_MODEL, image_plan(model_ref, family=family, config=config, platform_key=platform_key))


def video_model_part(name: str, model_ref: str, *, family: str, config: dict[str, Any], upscale: bool) -> JobDownloadPart:
    """Return the part for the video model *name* that loads from *model_ref* (with the upscale's files when it *upscale*s).

    Raises:
        ValueError: If the ``video_model_presets`` config entries of the family are malformed.
    """
    label = model_display_name(config, name)
    return JobDownloadPart(label, name, ROLE_MODEL, video_plan(model_ref, family=family, config=config, model_name=name, upscale=upscale))


def ltx_video_upscale_part(request: LtxVideoUpscaleRequest, config: dict[str, Any]) -> JobDownloadPart:
    """Return the part for the LTX model that refines an upscale of a video file, with the upscale's files.

    Raises:
        ValueError: If the ``video_model_presets`` config entries of the family are malformed.
    """
    return video_model_part(request.model_name, request.model_ref, family="ltx", config=config, upscale=True)


def audio_model_part(preset: AudioPreset, *, voice: bool = False) -> JobDownloadPart:
    """Return the part for the audio model *preset* an audio job generates with; *voice* when the take follows a recording, which also needs the cloning files."""
    return JobDownloadPart(preset.label, preset.id, ROLE_MODEL, audio_plan(preset, voice=voice))


def matte_part() -> JobDownloadPart:
    """Return the part for the background removal models of a transparent job."""
    return JobDownloadPart(MATTE_LABEL, MATTE_LICENCE_MODEL, ROLE_MATTE, matte_plan())


def seedvr2_tier_part(model_name: str, weights: SeedVR2Weights, models_dir: Path) -> JobDownloadPart:
    """Return the part for SeedVR2 tier *model_name* as *weights* run it, with its files under *models_dir*."""
    label = seedvr2_tier_label(model_name, weights)
    return JobDownloadPart(label, SEEDVR2_LICENCE_MODEL, ROLE_MODEL, seedvr2_plan(model_name, models_dir=models_dir, weights=weights))


def seedvr2_part(model_name: str | None) -> JobDownloadPart:
    """Return the part for SeedVR2 tier *model_name*; its targets are ``None`` when this platform has no SeedVR2 engine."""
    upscaler = get_seedvr2_upscaler()
    if upscaler is None:
        # The load reports that this platform has no SeedVR2 engine.
        return JobDownloadPart(str(model_name), SEEDVR2_LICENCE_MODEL, ROLE_MODEL, None)
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


def job_model_part(parts: Iterable[JobDownloadPart]) -> JobDownloadPart | None:
    """Return the part of the generation or upscale model among a job's *parts*, or ``None`` when the job has none (its plan could not be made)."""
    return next((part for part in parts if part.role == ROLE_MODEL), None)


def missing_targets(part: JobDownloadPart, present: Callable[[DownloadTarget], bool] = target_present_offline) -> tuple[DownloadTarget, ...] | None:
    """Return the targets of *part* that are not on disk, or ``None`` for a part that is unchecked (``targets=None``)."""
    if part.targets is None:
        return None
    return tuple(target for target in part.targets if not present(target))


def is_committed(missing: Sequence[DownloadTarget] | None, committed: Sequence[DownloadTarget]) -> bool:
    """Return whether the user has already been asked about a part: it still lacks something and every file it lacks is on its way.

    *missing* is :func:`missing_targets` of the part. *committed* holds the targets of running downloads, of queued and
    running jobs and of queued and running Models-page downloads. A lacking target is on its way when one of them covers it
    (same repo, revision and folder, and the committed files include the ones this target wants). A part that shares a
    file with a committed one (an encoder two models use) but lacks one of its own is not committed, so its own download
    is asked about. A part that is unchecked (``None``) is not, since what it would fetch is unknown.
    """
    return bool(missing) and all(any(_covers(done, target) for done in committed) for target in missing or ())


def _covers(committed: DownloadTarget, target: DownloadTarget) -> bool:
    """Return whether the download of *committed* fetches every file *target* wants: the same destination, with the same or more files."""
    if committed.key[:3] != target.key[:3]:
        return False
    if committed.allow_patterns is None:
        return True
    return target.allow_patterns is not None and set(target.allow_patterns) <= set(committed.allow_patterns)


def planned_targets(parts: Iterable[JobDownloadPart]) -> tuple[DownloadTarget, ...]:
    """Return every target of *parts*, in order; a part that is unchecked has none."""
    return tuple(target for part in parts for target in part.targets or ())


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
    model_id = enhancer_licence_model(tier) if tier in ENHANCER_LICENCE_TIERS else reference
    return JobDownloadPart(label, model_id, ROLE_ENHANCER, targets)


def _enhancer_tier(config: dict[str, Any], platform_key: str, repo: str, revision: str | None) -> str | None:
    """Return the configured enhancer tier whose model is ``(repo, revision)``, or ``None`` for a model no tier names."""
    return next((tier for tier, model in enhancer_tier_models(config, platform_key=platform_key).items() if model == (repo, revision)), None)
