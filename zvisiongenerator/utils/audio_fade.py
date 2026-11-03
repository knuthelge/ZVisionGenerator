"""Fade the end of a clip to silence. Pure: the input is left untouched and the result is a new array."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["fade_out"]


def fade_out(samples: NDArray[np.float32], rate: int, fade_seconds: float) -> NDArray[np.float32]:
    """Return *samples* with a linear fade to silence over their last *fade_seconds*.

    *samples* are float32 shaped ``(channels, frames)``; a fade longer than the clip fades the whole clip.
    """
    faded = np.array(samples, dtype=np.float32, copy=True)
    frames = min(int(round(fade_seconds * rate)), faded.shape[1])
    if frames > 0:
        faded[:, -frames:] *= np.linspace(1.0, 0.0, frames, dtype=np.float32)
    return faded
