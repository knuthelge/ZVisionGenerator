"""Compose the download targets a model needs from the platform planners, the Instant sources and the SeedVR2 tier files.

Every planner returns the targets still to be fetched, an empty tuple when nothing needs downloading, or ``None`` when
the plan cannot be made now (no connection to read the model's index); callers skip the ensure step for ``None``.
The platform planners live in :mod:`zvisiongenerator.backends` and are imported inside the functions, so importing
this module stays light.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.downloads.declaration import model_targets, role_targets
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.built_model import is_foreign_folder
from zvisiongenerator.utils.config import video_load_options
from zvisiongenerator.utils.ideogram4_instant import instant_downloads_dir, instant_source, instant_targets, is_built, is_instant_dir
from zvisiongenerator.utils.ltx_facts import ltx_version
from zvisiongenerator.utils.moss_build import is_moss_dir, is_turbo_dir, moss_source, moss_targets
from zvisiongenerator.utils.moss_build import is_built as is_moss_built
from zvisiongenerator.utils.paths import parse_huggingface_repo_reference
from zvisiongenerator.utils.seedvr2 import seedvr2_files

if TYPE_CHECKING:
    from zvisiongenerator.utils.audio_preset import AudioPreset
    from zvisiongenerator.utils.seedvr2 import SeedVR2Weights

__all__ = ["audio_plan", "built_model_plan", "enhancer_plan", "image_plan", "matte_plan", "seedvr2_plan", "video_model_downloaded", "video_plan"]


def image_plan(model_ref: str, *, family: str, platform_key: str) -> tuple[DownloadTarget, ...] | None:
    """Return the targets an image model needs, or ``None`` when its file list cannot be read now.

    Args:
        model_ref: The resolved model path or Hugging Face repo reference.
        family: The model family (``flux2_klein``, ``zimage``, ``krea2``, ``ideogram4``, ...).
        platform_key: The platform whose Instant sources apply (``sys.platform``).

    Returns:
        The targets, ``()`` when the model is a local folder or needs nothing, or ``None`` when offline.
    """
    built = built_model_plan(model_ref, platform_key=platform_key)
    if built is not None:
        return built
    from zvisiongenerator import backends

    return _offline_as_none(lambda: backends.image_download_plan(model_ref, family=family))


def built_model_plan(model_ref: str, *, platform_key: str) -> tuple[DownloadTarget, ...] | None:
    """Return the targets an unbuilt model built on first use needs, ``()`` once it is built, or ``None`` for any other model.

    A folder in the model's place that was not built by ziv needs nothing: the build refuses it before it downloads anything.
    """
    for plan in (_instant_plan(model_ref, platform_key=platform_key), _moss_plan(model_ref, platform_key=platform_key)):
        if plan is not None:
            return plan
    return None


def _instant_plan(model_ref: str, *, platform_key: str) -> tuple[DownloadTarget, ...] | None:
    """Return the targets an unbuilt Ideogram 4 Instant folder needs, ``()`` once it is built, or ``None`` for any other model.

    The targets are the base model's components at their pinned revision and the single-file transformer, which
    downloads into the hidden folder beside the model that the build resumes from.
    """
    if not is_instant_dir(model_ref):
        return None
    components, transformer = instant_targets(platform_key)
    folder = Path(model_ref)
    if is_built(folder, instant_source(platform_key)) or is_foreign_folder(folder):
        return ()
    return (components, replace(transformer, local_dir=str(instant_downloads_dir(folder))))


def _moss_plan(model_ref: str, *, platform_key: str) -> tuple[DownloadTarget, ...] | None:
    """Return the targets an unbuilt MOSS-Image folder needs, ``()`` once it is built, or ``None`` for any other model.

    The targets are the SFT repo's model files at their pinned revision and, for the Turbo folder, the LoRA file.
    """
    if not is_moss_dir(model_ref):
        return None
    sft, turbo = moss_targets(platform_key)
    folder = Path(model_ref).expanduser()
    if is_moss_built(folder, moss_source(platform_key)) or is_foreign_folder(folder):
        return ()
    return (sft, turbo) if is_turbo_dir(folder) else (sft,)


def video_plan(model_ref: str, *, family: str, config: dict[str, Any], model_name: str | None, platform_key: str, upscale: bool) -> tuple[DownloadTarget, ...] | None:
    """Return the targets a video model needs with the configured companions, or ``None`` when the plan cannot be made now.

    MiniMax H3 plans its repository, text encoder and turbo LoRA, plus the latent upscaler of the video upscale.

    The files come from the model metadata for *platform_key* (``sys.platform``); *config* supplies the settings.

    Raises:
        ValueError: If a video setting of the family in *config* is malformed.
    """
    from zvisiongenerator import backends

    options = video_load_options(family, config, model_name, platform_key=platform_key, upscale=upscale)
    if family == "h3":
        lora = {"lora": options["lora"]} if "lora" in options else {}  # macOS lists the turbo LoRA from the load options
        return _offline_as_none(lambda: backends.h3_download_plan(model_ref, options["preset"], upscale=upscale, **lora))
    return _offline_as_none(
        lambda: backends.video_download_plan(
            model_ref,
            family=family,
            transformer_source=options["transformer_source"],
            latent_upsampler=options.get("latent_upsampler"),
            upscale=upscale,
            repo_ignore=options["repo_ignore"],
            text_encoder_bundled=ltx_version(model_name) is not None,
            mlx_packs=options["mlx_packs"],
            mlx_text_encoder=options.get("mlx_text_encoder"),
        )
    )


def audio_plan(preset: AudioPreset, *, voice: bool = False) -> tuple[DownloadTarget, ...]:
    """Return one target per repository the audio *preset* loads, at the pinned commit, in the order the preset lists them.

    A take that follows a voice recording (*voice*) also needs the preset's ``voice_files``, which come last; a built-in voice
    does not. The plan comes from the preset alone, so it needs no connection and is never ``None``.
    """
    files = (*preset.files, *preset.voice_files) if voice else preset.files
    return tuple(DownloadTarget(item.repo_id, item.revision, item.patterns) for item in files)


def video_model_downloaded(
    model_ref: str,
    *,
    family: str,
    config: dict[str, Any],
    model_name: str | None,
    platform_key: str,
    target_present: Callable[[DownloadTarget], bool] | None = None,
) -> bool:
    """Return whether every download of the video model's plan on this platform is on disk, judged without the network.

    A plan that cannot be made now counts as not downloaded, and so does a reference that is neither a Hugging Face repo
    nor an existing local folder (nothing could be downloaded for it).

    Args:
        model_ref: The resolved model path or Hugging Face repo reference.
        family: The model family (``ltx`` or ``h3``).
        config: The app config.
        model_name: The name the model is picked by (``ltx-4``).
        platform_key: The platform whose declared files apply (``sys.platform``).
        target_present: Tells whether one target is fully on disk; defaults to the offline cache check.

    Raises:
        ValueError: If a video setting of the family in *config* is malformed.
    """
    if not Path(model_ref).expanduser().is_dir() and parse_huggingface_repo_reference(model_ref) is None:
        return False  # A local path that is no folder (deleted or never made) has nothing to download.
    if target_present is None:
        from zvisiongenerator.downloads.presence import target_present_offline

        target_present = target_present_offline
    plan = video_plan(model_ref, family=family, config=config, model_name=model_name, platform_key=platform_key, upscale=False)
    return plan is not None and all(target_present(target) for target in plan)


def seedvr2_plan(model: str, *, platform_key: str, models_dir: Path, weights: SeedVR2Weights) -> tuple[DownloadTarget, ...]:
    """Return the targets the SeedVR2 *model* still needs on *platform_key*: none once its weights are in place, else one per declared repo.

    Engines that keep a stored copy download the tier's files into the Hugging Face cache (the load quantizes
    them); the others download straight into the tier's weights folder.
    """
    from zvisiongenerator.seedvr2_loader import seedvr2_downloaded, seedvr2_weights_dir

    if seedvr2_downloaded(model, models_dir, weights, platform_key=platform_key):
        return ()
    local_dir = None if weights.stores_copy else str(seedvr2_weights_dir(model, models_dir, weights))
    names_by_repo: dict[str, list[str]] = {}
    for repo, name in seedvr2_files(model, platform_key):
        names_by_repo.setdefault(repo, []).append(name)
    return tuple(DownloadTarget(repo, None, tuple(names), local_dir=local_dir) for repo, names in names_by_repo.items())


def enhancer_plan(repo: str, revision: str | None) -> tuple[DownloadTarget, ...]:
    """Return the targets a prompt enhancer tier needs on this platform (``()`` for a local folder)."""
    from zvisiongenerator import backends

    return backends.enhancer_download_plan(repo, revision)


def matte_plan() -> tuple[DownloadTarget, ...]:
    """Return the two pinned targets the background removal chain needs (BiRefNet, then ViTMatte), declared for this platform."""
    pairs = model_targets("matte", sys.platform)
    return (*role_targets(pairs, "birefnet"), *role_targets(pairs, "vitmatte"))


def _offline_as_none(plan: Callable[[], tuple[DownloadTarget, ...]]) -> tuple[DownloadTarget, ...] | None:
    """Run *plan*, returning ``None`` when the model's file list cannot be read (no connection, or no cached index)."""
    from huggingface_hub.utils import httpx
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

    try:
        return plan()
    except LookupError, httpx.TransportError, LocalEntryNotFoundError, OfflineModeIsEnabled:
        return None
