"""Name the cache repos that models built on first use are made from.

MOSS-Image and Ideogram 4 Instant load from folders in the models directory, built by linking files out of the
Hugging Face cache. Their aliases resolve to those folders, not to repos, so the source repos are named here.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.ideogram4_instant import INSTANT_DIR_NAME, instant_downloads_dir, instant_source
from zvisiongenerator.utils.moss_build import SFT_DIR_NAME, TURBO_DIR_NAME, moss_source
from zvisiongenerator.utils.platform import resolve_alias

__all__ = ["BUILT_MODEL_NAMES", "built_model_downloads", "built_model_source_repos", "built_model_sources"]

# The aliases (and model folder names) of the models built on first use.
BUILT_MODEL_NAMES = (TURBO_DIR_NAME, SFT_DIR_NAME, INSTANT_DIR_NAME)


def built_model_source_repos(app_config: dict[str, Any], *, platform_key: str) -> frozenset[str]:
    """Return the cache repo ids that the built-on-first-use models available on *platform_key* are made from.

    A model counts when its alias resolves for the platform: ``moss`` uses the SFT repo and the Turbo LoRA repo,
    ``moss-sft`` the SFT repo, and the Ideogram 4 Instant alias its components repo (the single-file transformer is
    stored beside the model, not in the cache). A source block that is invalid is skipped with a warning.
    """
    aliases = app_config.get("model_aliases", {})
    repos: set[str] = set()
    for name in BUILT_MODEL_NAMES:
        if not _has_alias(aliases, name, platform_key):
            continue
        try:
            repos.update(built_model_sources(name, app_config, platform_key=platform_key))
        except ValueError as exc:
            warnings.warn(f"Could not read the sources of a model built on first use: {exc}", stacklevel=2)
    return frozenset(repos)


def built_model_sources(name: str, app_config: dict[str, Any], *, platform_key: str) -> tuple[str, ...]:
    """Return the cache repo ids that the model built on first use called *name* is made from, or ``()`` for any other name.

    Raises:
        ValueError: If the source block of the model in *app_config* is invalid.
    """
    if name == TURBO_DIR_NAME:
        source = moss_source(app_config)
        return (source.sft_repo, source.turbo_repo)
    if name == SFT_DIR_NAME:
        return (moss_source(app_config).sft_repo,)
    if name == INSTANT_DIR_NAME:
        return (instant_source(app_config, platform_key).components_repo,)
    return ()


def built_model_downloads(name: str, models_dir: Path) -> tuple[Path, ...]:
    """Return the hidden folders in *models_dir* that hold files the build of the model called *name* downloaded outside the cache.

    Ideogram 4 Instant downloads its single-file transformer next to the model, where an interrupted download resumes
    and the finished build removes it. MOSS-Image downloads only into the cache. Any other name has none. Only the
    folders that exist are returned.
    """
    folders = (instant_downloads_dir(models_dir / name),) if name == INSTANT_DIR_NAME else ()
    return tuple(folder for folder in folders if folder.is_dir())


def _has_alias(aliases: dict[str, Any], name: str, platform_key: str) -> bool:
    """Return whether alias *name* is defined and resolves on *platform_key*."""
    if name not in aliases:
        return False
    try:
        resolve_alias(aliases[name], platform_key)
    except ValueError:
        return False
    return True
