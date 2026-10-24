"""Resolve the model, steps, size and length a Web UI video job runs at: the same answer for the job and for its memory estimate."""

from __future__ import annotations

import contextlib
import sys
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from zvisiongenerator.core.video_family import ANY_FRAMES
from zvisiongenerator.utils.alignment import aligned_video_dims
from zvisiongenerator.utils.config import resolve_video_defaults, validate_video_size_choice, video_family
from zvisiongenerator.utils.paths import resolve_model_path
from zvisiongenerator.utils.video_model_detect import VideoModelInfo, detect_video_model


@dataclass(frozen=True)
class VideoClip:
    """The steps and the aligned size and frame count a text-to-video or image-to-video job runs at."""

    steps: int
    width: int
    height: int
    num_frames: int


def clip_defaults(
    app_config: dict[str, Any],
    model_info: VideoModelInfo,
    *,
    ratio: str | None = None,
    size: str | None = None,
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
) -> dict[str, Any]:
    """Resolve a video job's steps, size and frame count: what the form gave, else the model's presets and the config defaults."""
    cli_overrides = {key: value for key, value in {"ratio": ratio, "size": size, "steps": steps, "width": width, "height": height, "num_frames": frames}.items() if value is not None}
    return resolve_video_defaults(model_info.family, app_config, cli_overrides)


def resolve_video_model(app_config: dict[str, Any], model_name: str) -> tuple[str, VideoModelInfo]:
    """Return the resolved path and the detected info of a video model.

    Raises:
        ValueError: If the model's family is not supported.
    """
    resolved_model = resolve_model_path(model_name, aliases=app_config.get("model_aliases", {}), platform_key=sys.platform)
    model_info = detect_video_model(resolved_model)
    if model_info.family == "unknown":
        raise ValueError(f"Could not detect a supported video model for '{model_name}'.")
    return resolved_model, model_info


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Suppress warnings inside the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def resolve_video_clip(
    app_config: dict[str, Any],
    model_info: VideoModelInfo,
    *,
    ratio: str,
    size: str,
    steps: int | None,
    width: int | None,
    height: int | None,
    frames: int | None,
    upscale: bool,
    quiet: bool = False,
) -> VideoClip:
    """Resolve the steps, size and length of a clip exactly as its job runs it: the configured presets, then the model's alignment.

    A job and the memory estimate of the same request both come through here, so they cannot disagree.

    Args:
        quiet: Suppress the warnings that report alignment corrections (the estimate reports them in its response).

    Raises:
        ValueError: If the ratio and size are not in the size table, or the aligned size is under 64 pixels.
    """
    validate_video_size_choice(app_config, ratio, size)
    defaults = clip_defaults(app_config, model_info, ratio=ratio, size=size, steps=steps, width=width, height=height, frames=frames)
    family = video_family(app_config, model_info.family)
    with _quiet() if quiet else contextlib.nullcontext():
        aligned_width, aligned_height, aligned_frames = aligned_video_dims(
            defaults["width"],
            defaults["height"],
            defaults["num_frames"],
            upscale=upscale,
            resolution_alignment=model_info.resolution_alignment,
            frame_rule=family.frame_rule if family is not None else ANY_FRAMES,
            family=model_info.family.upper(),
        )
    if aligned_width < 64 or aligned_height < 64:
        raise ValueError(f"Resolved dimensions {aligned_width}x{aligned_height} are too small (minimum 64x64)")
    return VideoClip(steps=defaults["steps"], width=aligned_width, height=aligned_height, num_frames=aligned_frames)


def require_upscale_factor(upscale: int | None) -> None:
    """Raise ``ValueError`` unless *upscale* is unset or 2, the only factor the LTX spatial upscaler has."""
    if upscale is not None and upscale != 2:
        raise ValueError("Upscale only supports factor 2 (LTX spatial upscaler).")
