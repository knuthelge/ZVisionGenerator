"""Fade the end of a clip to silence. Pure: the input is left untouched and the result is a new array."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["fade_out"]


def fade_out(samples: NDArray[np.float32], rate: int, fade_seconds: float) -> NDArray[np.float32]:
    """Return *samples* with a linear fade to silence over their last *fade_seconds*.

    A fade longer than the clip fades the whole clip. A *fade_seconds* of 0 returns an unchanged copy.

    Args:
        samples: Float32 samples shaped ``(channels, frames)``.
        rate: Samples per second.
        fade_seconds: The length of the fade in seconds.

    Returns:
        A new float32 array of the same shape whose last frame is silent (when the fade is longer than 0 frames).

    Raises:
        ValueError: If the samples are not a ``(channels, frames)`` array, the rate is below 1 or the fade is negative.
    """
    if samples.ndim != 2:
        raise ValueError("fade_out needs samples shaped (channels, frames).")
    if rate < 1:
        raise ValueError("fade_out needs a sample rate of at least 1.")
    if fade_seconds < 0:
        raise ValueError("fade_out needs a fade length of 0 or more seconds.")
    faded = np.array(samples, dtype=np.float32, copy=True)
    frames = min(int(round(fade_seconds * rate)), faded.shape[1])
    if frames > 0:
        faded[:, -frames:] *= np.linspace(1.0, 0.0, frames, dtype=np.float32)
    return faded
