"""Build the Ideogram 4 Instant model folder on first use.

The folder (``<models dir>/ideo``, named like its alias) is assembled in a hidden partial
folder: the base model's text encoder, VAE, tokenizer and scheduler are hard-linked from the Hugging Face cache,
and the transformer is converted from the single-file BF16 checkpoint, which downloads into a hidden folder next to
the model (an interrupted download resumes there) and is deleted once the build is complete. A build record written
last marks the folder finished; a change to the pinned sources in config rebuilds it.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zvisiongenerator.converters.ideogram4_instant import write_transformer
from zvisiongenerator.image_model_loader import LOADING_PHASE
from zvisiongenerator.utils.built_model import BUILDING_PHASE, build_in_partial
from zvisiongenerator.utils.file_links import link_tree
from zvisiongenerator.utils.ideogram4_instant import (
    COMPONENT_PATTERNS,
    LICENSE_FILES,
    LINKED_COMPONENTS,
    InstantSource,
    instant_downloads_dir,
    instant_source,
    is_built,
    is_instant_dir,
    write_build_record,
)

__all__ = ["build_instant_model", "ensure_instant_model", "instant_model_index"]

Downloader = Callable[[InstantSource, Path], Path]


def ensure_instant_model(
    model_path: str,
    config: dict[str, Any],
    *,
    platform_key: str,
    on_phase: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> bool:
    """Build the Ideogram 4 Instant folder at *model_path* unless it is already built from the configured sources.

    Any other *model_path* is left alone.

    Args:
        model_path: The resolved model path.
        config: The app config, whose ``ideogram4_instant`` block pins the sources.
        platform_key: Picks the platform's components repo and transformer format.
        on_phase: Receives :data:`BUILDING_PHASE` when a build starts, then :data:`LOADING_PHASE` once it is done.
        cancelled: Checked during the conversion; a true result stops the build.

    Returns:
        Whether the model is ready (``False`` only when *cancelled* stopped the build).
    """
    if not is_instant_dir(model_path):
        return True
    source = instant_source(config, platform_key)
    target = Path(model_path)
    if is_built(target, source):
        return True
    if on_phase is not None:
        on_phase(BUILDING_PHASE)
    built = build_instant_model(source, target, cancelled=cancelled)
    if built and on_phase is not None:
        on_phase(LOADING_PHASE)
    return built


def build_instant_model(
    source: InstantSource,
    target: Path,
    *,
    cancelled: Callable[[], bool] | None = None,
    download_components: Downloader | None = None,
    download_transformer: Downloader | None = None,
) -> bool:
    """Build the Ideogram 4 Instant folder at *target* from *source*, replacing any older build.

    Args:
        source: The pinned transformer file, components repo and transformer format.
        target: The model folder to create.
        cancelled: Checked before each converted tensor; a true result discards the partial folder.
        download_components: Returns the local folder of the components repo (default: the Hugging Face cache).
        download_transformer: Downloads the transformer file into the given folder and returns its path.

    Returns:
        Whether the folder is in place (``False`` when *cancelled* stopped the build).
    """
    download_components = download_components or _download_components
    download_transformer = download_transformer or _download_transformer
    downloads = instant_downloads_dir(target)

    def assemble(partial: Path) -> bool:
        components = download_components(source, downloads)
        transformer_file = download_transformer(source, downloads)
        partial.mkdir()
        _link_components(components, partial)
        base_config = json.loads((components / "transformer" / "config.json").read_text(encoding="utf-8"))
        return write_transformer(transformer_file, partial / "transformer", source.transformer_format, base_config, cancelled)

    if not build_in_partial(target, assemble, lambda partial: write_build_record(partial, source)):
        return False
    shutil.rmtree(downloads, ignore_errors=True)
    return True


def instant_model_index(base_index: dict[str, Any]) -> dict[str, Any]:
    """Return the base model's ``model_index.json`` for Instant: no unconditional transformer, no copy-specific path."""
    index = {key: value for key, value in base_index.items() if key != "_name_or_path"}
    index["unconditional_transformer"] = [None, None]
    return index


def _link_components(components: Path, target: Path) -> None:
    """Hard-link the reused components and license files of *components* into *target* and write Instant's ``model_index.json``."""
    for name in LINKED_COMPONENTS:
        link_tree(components / name, target / name)
    for name in LICENSE_FILES:
        if (components / name).is_file():
            link_tree(components / name, target / name)
    base_index = json.loads((components / "model_index.json").read_text(encoding="utf-8"))
    (target / "model_index.json").write_text(json.dumps(instant_model_index(base_index), indent=2) + "\n", encoding="utf-8")


def _download_components(source: InstantSource, _downloads: Path) -> Path:
    """Return the components repo's snapshot in the Hugging Face cache, downloading the reused parts."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(source.components_repo, revision=source.components_revision, allow_patterns=list(COMPONENT_PATTERNS)))


def _download_transformer(source: InstantSource, downloads: Path) -> Path:
    """Download the transformer file into *downloads* (resuming an interrupted download) and return its path."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(source.transformer_repo, source.transformer_file, revision=source.transformer_revision, local_dir=downloads))
