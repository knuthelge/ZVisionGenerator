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
