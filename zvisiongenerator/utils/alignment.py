"""Round image and video dimensions to backend-supported alignments."""

from __future__ import annotations

import warnings


def round_to_alignment(value: int, alignment: int = 16) -> int:
    """Round *value* up to the nearest multiple of *alignment*."""
    return (value + alignment - 1) // alignment * alignment


def align_resolution(width: int, height: int, divisor: int, label: str = "Video") -> tuple[int, int]:
    """Round width and height to the nearest multiple of *divisor*."""
    half = divisor // 2
    aligned_w = ((width + half) // divisor) * divisor
    aligned_h = ((height + half) // divisor) * divisor
    if aligned_w != width or aligned_h != height:
        warnings.warn(
            f"{label} requires dimensions divisible by {divisor}. Adjusted {width}x{height} -> {aligned_w}x{aligned_h}",
            stacklevel=2,
        )
    return aligned_w, aligned_h


def align_ltx_frames(frames: int, alignment: int = 8) -> int:
    """Round a frame count to the nearest valid ``alignment * k + 1`` value."""
    if alignment <= 0:
        return frames
    k = max(1, round((frames - 1) / alignment))
    aligned = alignment * k + 1
    if aligned != frames:
        warnings.warn(f"Video model requires frames = {alignment}k+1. Adjusted {frames} -> {aligned}", stacklevel=2)
    return aligned


def aligned_video_dims(
    width: int,
    height: int,
    frames: int,
    *,
    upscale: bool,
    resolution_alignment: int,
    frame_alignment: int,
    family: str,
) -> tuple[int, int, int]:
    """Round a video job's size and frame count to what its model accepts, warning about each correction.

    An upscaled clip uses 64-alignment so its half-size stage stays 32-aligned; otherwise the model's own
    alignment applies.

    Args:
        width: Requested width in pixels.
        height: Requested height in pixels.
        frames: Requested frame count.
        upscale: Whether the clip is upscaled.
        resolution_alignment: The model's size alignment.
        frame_alignment: The model's frame alignment (frames are ``alignment * k + 1``).
        family: The model family label used in the warning.

    Returns:
        The aligned ``(width, height, frames)``.
    """
    aligned_w, aligned_h = align_resolution(width, height, 64 if upscale else resolution_alignment, family)
    return aligned_w, aligned_h, align_ltx_frames(frames, frame_alignment)
