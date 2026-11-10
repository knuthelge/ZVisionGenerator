"""Turn a time in a clip into the frame it lies in, and that frame into the time ffmpeg seeks to.

The Web UI's player applies the same rule in ``frontend/src/lib/state/framePoint.ts``; both are kept equal by one shared fixture.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zvisiongenerator.utils.video_probe import VideoProbe

__all__ = ["frame_at", "frame_seek_seconds", "frame_total"]

# Absorbs the float error of a time that was computed as ``start + frame / fps``, so it lands on its own frame.
_FRAME_EPSILON = 1e-3


def frame_at(seconds: float, *, start: float, fps: float, frames: int) -> int:
    """Return the index of the frame a time lies in, clamped to the clip.

    Args:
        seconds: A time on the container's clock.
        start: Seconds from the file start to the first video frame.
        fps: The constant frame rate.
        frames: How many frames the clip has (at least 1).

    Raises:
        ValueError: If *fps* is not positive or *frames* is less than 1.
    """
    if fps <= 0 or frames < 1:
        raise ValueError("A frame needs a frame rate above zero and at least one frame.")
    return min(max(math.floor((seconds - start) * fps + _FRAME_EPSILON), 0), frames - 1)


def frame_seek_seconds(frame: int, *, start: float, fps: float) -> float:
    """Return the time to give ffmpeg's ``-ss`` so that *frame* is the first frame it decodes: half a frame before it, never negative."""
    return max(0.0, start + (frame - 0.5) / fps)


def frame_total(probe: VideoProbe) -> int | None:
    """Return how many frames a clip has: the counted or recorded number, else its length times its rate; ``None`` when neither is known."""
    if probe.frame_count:
        return probe.frame_count
    if probe.duration and probe.fps:
        return round(probe.duration * probe.fps)
    return None
