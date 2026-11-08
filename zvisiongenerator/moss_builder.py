"""Build a MOSS-Image model folder on first use.

Each alias (``<models dir>/moss`` and ``<models dir>/moss-sft``) is assembled in a hidden partial folder: the SFT
repo's components are hard-linked from the Hugging Face cache, the folder's ``model_index.json`` is written with the
alias's ``is_distilled`` flag and, for Turbo, the 4-step LoRA is linked in beside them. A build record written last
marks the folder finished; a change to the pinned sources in the model metadata rebuilds it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.image_model_loader import LOADING_PHASE
from zvisiongenerator.utils.built_model import BUILDING_PHASE, build_in_partial
from zvisiongenerator.utils.file_links import link_tree
from zvisiongenerator.utils.moss_build import (
    LINKED_COMPONENTS,
    TURBO_LORA_NAME,
    MossSource,
    is_built,
    is_moss_dir,
    is_turbo_dir,
    moss_source,
    moss_targets,
    write_build_record,
)

__all__ = ["build_moss_model", "ensure_moss_model", "moss_model_index"]

Downloader = Callable[[MossSource], Path]


def ensure_moss_model(
    model_path: str,
    *,
    platform_key: str,
    on_phase: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> bool:
    """Build the MOSS-Image folder at *model_path* unless it is already built from the pinned sources.

    Any other *model_path* is left alone.

    Args:
        model_path: The resolved model path.
        platform_key: The platform whose declared sources apply (``sys.platform``).
        on_phase: Receives :data:`BUILDING_PHASE` when a build starts, then :data:`LOADING_PHASE` once it is done.
        cancelled: Checked between the build's steps; a true result stops the build.

    Returns:
        Whether the model is ready (``False`` only when *cancelled* stopped the build).

    Raises:
        RuntimeError: If *model_path* holds something ziv did not build (see :func:`~zvisiongenerator.utils.built_model.build_in_partial`).
    """
    if not is_moss_dir(model_path):
        return True
    source = moss_source(platform_key)
    target = Path(model_path).expanduser()
    if is_built(target, source):
        return True
    if on_phase is not None:
        on_phase(BUILDING_PHASE)
    sft, _turbo = moss_targets(platform_key)
    built = build_moss_model(source, target, turbo=is_turbo_dir(target), cancelled=cancelled, download_sft=_sft_downloader(sft))
    if built and on_phase is not None:
        on_phase(LOADING_PHASE)
    return built


def build_moss_model(
    source: MossSource,
    target: Path,
    *,
    turbo: bool,
    download_sft: Downloader,
    cancelled: Callable[[], bool] | None = None,
    download_lora: Downloader | None = None,
) -> bool:
    """Build the MOSS-Image folder at *target* from *source*, replacing any older build.

    Args:
        source: The pinned SFT repo and Turbo LoRA.
        target: The model folder to create.
        turbo: Whether to build the Turbo alias (distilled flag set, LoRA linked in) rather than SFT.
        download_sft: Returns the local folder of the SFT repo.
        cancelled: Checked between the steps; a true result discards the partial folder.
        download_lora: Returns the local Turbo LoRA file (default: the Hugging Face cache); only used when *turbo*.

    Returns:
        Whether the folder is in place (``False`` when *cancelled* stopped the build).
    """
    download_lora = download_lora or _download_lora

    def stopped() -> bool:
        """Return whether the caller asked to stop."""
        return cancelled is not None and cancelled()

    def assemble(partial: Path) -> bool:
        snapshot = download_sft(source)
        if stopped():
            return False
        lora = download_lora(source) if turbo else None
        if stopped():
            return False
        partial.mkdir()
        _link_components(snapshot, partial, turbo=turbo)
        if lora is not None:
            link_tree(lora, partial / TURBO_LORA_NAME)
        return True

    return build_in_partial(target, assemble, lambda partial: write_build_record(partial, source, turbo=turbo))


def moss_model_index(sft_index: dict[str, Any], *, turbo: bool) -> dict[str, Any]:
    """Return the SFT repo's ``model_index.json`` for an alias: no copy-specific path, ``is_distilled`` set to *turbo*."""
    index = {key: value for key, value in sft_index.items() if key != "_name_or_path"}
    index["is_distilled"] = turbo
    return index


def _link_components(snapshot: Path, target: Path, *, turbo: bool) -> None:
    """Hard-link the components and license of *snapshot* into *target* and write the alias's ``model_index.json``."""
    for name in LINKED_COMPONENTS:
        link_tree(snapshot / name, target / name)
    if (snapshot / "LICENSE").is_file():
        link_tree(snapshot / "LICENSE", target / "LICENSE")
    sft_index = json.loads((snapshot / "model_index.json").read_text(encoding="utf-8"))
    (target / "model_index.json").write_text(json.dumps(moss_model_index(sft_index, turbo=turbo), indent=2) + "\n", encoding="utf-8")


def _sft_downloader(sft: DownloadTarget) -> Downloader:
    """Return the downloader of the SFT repo: its snapshot in the Hugging Face cache, with the model files the *sft* target declares (never its scripts)."""

    def download(source: MossSource) -> Path:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(source.sft_repo, revision=source.sft_revision, allow_patterns=list(sft.allow_patterns), ignore_patterns=list(sft.ignore_patterns)))

    return download


def _download_lora(source: MossSource) -> Path:
    """Return the Turbo LoRA file in the Hugging Face cache, downloading it."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(source.turbo_repo, source.turbo_file, revision=source.turbo_revision))
