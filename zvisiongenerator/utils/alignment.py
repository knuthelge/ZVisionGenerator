"""Round image and video dimensions to backend-supported alignments."""

from __future__ import annotations

import warnings

from zvisiongenerator.core.video_family import FrameRule


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


def align_frames(frames: int, rule: FrameRule) -> int:
    """Move a frame count to the nearest valid ``alignment * k + offset`` count of *rule* (``k >= 1``).

    The rule's ``round`` picks the direction: ``"nearest"`` rounds halves to even, ``"up"`` takes the next valid count.
    There is no upper limit.
    """
    quotient = (frames - rule.offset) / rule.alignment
    k = -((rule.offset - frames) // rule.alignment) if rule.round == "up" else round(quotient)
    return rule.alignment * max(1, k) + rule.offset


def _align_frames_warning(frames: int, rule: FrameRule) -> int:
    """Return *frames* aligned to *rule*, warning when the count changed (the warning is attributed to the caller of this helper)."""
    aligned = align_frames(frames, rule)
    if aligned != frames:
        warnings.warn(f"Video model requires frames = {rule.alignment}k+{rule.offset}. Adjusted {frames} -> {aligned}", stacklevel=2)
    return aligned


def aligned_video_dims(
    width: int,
    height: int,
    frames: int,
    *,
    upscale: bool,
    resolution_alignment: int,
    frame_rule: FrameRule,
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
        frame_rule: The frame counts the model accepts.
        family: The model family label used in the warning.

    Returns:
        The aligned ``(width, height, frames)``.
    """
    aligned_w, aligned_h = align_resolution(width, height, 64 if upscale else resolution_alignment, family)
    return aligned_w, aligned_h, _align_frames_warning(frames, frame_rule)
