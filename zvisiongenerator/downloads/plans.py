"""Compose the download targets a model needs from the platform planners, the Instant sources and the SeedVR2 tier files.

Every planner returns the targets still to be fetched, an empty tuple when nothing needs downloading, or ``None`` when
the plan cannot be made now (no connection to read the model's index); callers skip the ensure step for ``None``.
The platform planners live in :mod:`zvisiongenerator.backends` and are imported inside the functions, so importing
this module stays light.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.ideogram4_instant import COMPONENT_PATTERNS, instant_downloads_dir, instant_source, is_built, is_instant_dir

if TYPE_CHECKING:
    from zvisiongenerator.utils.seedvr2 import SeedVR2Weights

__all__ = ["enhancer_plan", "image_plan", "instant_plan", "seedvr2_plan", "video_plan"]


def image_plan(model_ref: str, *, family: str, config: dict[str, Any] | None, platform_key: str) -> tuple[DownloadTarget, ...] | None:
    """Return the targets an image model needs, or ``None`` when its file list cannot be read now.

    Args:
        model_ref: The resolved model path or Hugging Face repo reference.
        family: The model family (``flux2_klein``, ``zimage``, ``krea2``, ``ideogram4``, ...).
        config: The app config; when given, an Ideogram 4 Instant folder is planned from its pinned sources first.
        platform_key: The platform whose Instant sources apply (``sys.platform``).

    Returns:
        The targets, ``()`` when the model is a local folder or needs nothing, or ``None`` when offline.
    """
    if config is not None:
        instant = instant_plan(model_ref, config=config, platform_key=platform_key)
        if instant is not None:
            return instant
    from zvisiongenerator import backends

    return _offline_as_none(lambda: backends.image_download_plan(model_ref, family=family))


def instant_plan(model_ref: str, *, config: dict[str, Any], platform_key: str) -> tuple[DownloadTarget, ...] | None:
    """Return the targets an unbuilt Ideogram 4 Instant folder needs, ``()`` once it is built, or ``None`` for any other model.

    The targets are the base model's components at their pinned revision and the single-file transformer, which
    downloads into the hidden folder beside the model that the build resumes from.

    Raises:
        ValueError: If the ``ideogram4_instant`` block of *config* is invalid.
    """
    if not is_instant_dir(model_ref):
        return None
    source = instant_source(config, platform_key)
    folder = Path(model_ref)
    if is_built(folder, source):
        return ()
    return (
        DownloadTarget(source.components_repo, source.components_revision, COMPONENT_PATTERNS),
        DownloadTarget(source.transformer_repo, source.transformer_revision, (source.transformer_file,), local_dir=str(instant_downloads_dir(folder))),
    )


def video_plan(model_ref: str, *, family: str, config: dict[str, Any], model_name: str | None, upscale: bool) -> tuple[DownloadTarget, ...] | None:
    """Return the targets a video model needs with the configured companions, or ``None`` when the plan cannot be made now.

    MiniMax H3 plans its repository, text encoder and turbo LoRA, plus the latent upscaler of the video upscale.

    Raises:
        ValueError: If the ``video_model_presets`` config entries of the family are malformed.
    """
    from zvisiongenerator import backends
    from zvisiongenerator.utils.config import ltx_version, video_load_options

    options = video_load_options(family, config, model_name, upscale=upscale)
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
            repo_ignore=options.get("repo_ignore", ()),
            text_encoder_bundled=ltx_version(config, model_name) is not None,
        )
    )


def seedvr2_plan(model: str, *, models_dir: Path, weights: SeedVR2Weights) -> tuple[DownloadTarget, ...]:
    """Return the targets the SeedVR2 *model* still needs: none once its weights are in place, else one per repo.

    Engines that keep a stored copy download the tier's files into the Hugging Face cache (the load quantizes
    them); the others download straight into the tier's weights folder.
    """
    from zvisiongenerator.seedvr2_loader import seedvr2_downloaded, seedvr2_weights_dir

    if seedvr2_downloaded(model, models_dir, weights):
        return ()
    local_dir = None if weights.stores_copy else str(seedvr2_weights_dir(model, models_dir, weights))
    names_by_repo: dict[str, list[str]] = {}
    for repo, name in weights.files[model]:
        names_by_repo.setdefault(repo, []).append(name)
    return tuple(DownloadTarget(repo, None, tuple(names), local_dir=local_dir) for repo, names in names_by_repo.items())


def enhancer_plan(repo: str, revision: str | None) -> tuple[DownloadTarget, ...]:
    """Return the targets a prompt enhancer tier needs on this platform (``()`` for a local folder)."""
    from zvisiongenerator import backends

    return backends.enhancer_download_plan(repo, revision)


def _offline_as_none(plan: Callable[[], tuple[DownloadTarget, ...]]) -> tuple[DownloadTarget, ...] | None:
    """Run *plan*, returning ``None`` when the model's file list cannot be read (no connection, or no cached index)."""
    import httpx
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

    try:
        return plan()
    except LookupError, httpx.TransportError, LocalEntryNotFoundError, OfflineModeIsEnabled:
        return None
