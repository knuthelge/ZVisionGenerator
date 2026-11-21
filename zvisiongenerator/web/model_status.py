"""Tell whether a model is downloaded, or has nothing left to download."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zvisiongenerator.downloads.plans import built_model_plan, video_model_downloaded
from zvisiongenerator.downloads.presence import target_present_offline
from zvisiongenerator.downloads.target import DownloadTarget
from zvisiongenerator.utils.model_files import find_local_model_dir


def model_downloaded(
    resolved_path: str,
    *,
    kind: str,
    family: str | None = None,
    model_name: str | None = None,
    app_config: dict[str, Any],
    find_local_dir: Callable[[str], Path | None] = find_local_model_dir,
    target_present: Callable[[DownloadTarget], bool] = target_present_offline,
) -> bool:
    """Return whether the model is on disk, from what is on disk alone.

    MiniMax H3 is downloaded when every target of its plan on this platform is on disk. An LTX model is downloaded when its
    folder resolves and every target of its plan is on disk. Any other model is downloaded when its folder resolves.

    Args:
        resolved_path: The model's local directory or HuggingFace repo id.
        kind: ``"image"`` or ``"video"``.
        family: A video model's family (``"ltx"`` or ``"h3"``); ``None`` is LTX.
        model_name: The name the model is picked by (``ltx-4``).
        app_config: The loaded config: its video settings shape the download plan of a video model.
        find_local_dir: Resolver for fully downloaded model directories.
        target_present: Tells whether one download target is fully on disk.
    """
    if kind == "video" and family == "h3":
        return _video_downloaded(resolved_path, "h3", model_name, app_config, target_present)
    model_dir = find_local_dir(resolved_path)
    if kind == "video" and family in (None, "ltx"):  # The folder check covers a local folder (its plan is empty); the plan covers the files a Hub reference still needs.
        return model_dir is not None and _video_downloaded(resolved_path, "ltx", model_name, app_config, target_present)
    return model_dir is not None


def nothing_to_download(
    resolved_path: str,
    *,
    downloaded: bool,
    target_present: Callable[[DownloadTarget], bool] = target_present_offline,
) -> bool:
    """Return whether a model has nothing left to fetch: it is *downloaded*, or it is built on first use and all its sources are on disk.

    A model built on first use (Ideogram 4 Instant, MOSS-Image) that is not built yet is not downloaded, but its first job
    builds it from the sources without a download. Every surface that tells a user what a model still costs asks this one rule.

    Args:
        resolved_path: The model's local directory or HuggingFace repo id.
        downloaded: The model's plain :func:`model_downloaded` answer.
        target_present: Tells whether one download target is fully on disk.
    """
    if downloaded:
        return True
    sources = built_model_plan(resolved_path, platform_key=sys.platform)
    return sources is not None and all(target_present(target) for target in sources)


def _video_downloaded(resolved_path: str, family: str, model_name: str | None, app_config: dict[str, Any], target_present: Callable[[DownloadTarget], bool]) -> bool:
    """Return whether every download of the video model's plan on this platform is on disk (a plan that cannot be made counts as not downloaded)."""
    return video_model_downloaded(resolved_path, family=family, config=app_config, model_name=model_name or resolved_path, platform_key=sys.platform, target_present=target_present)
