"""Bring a clip that peaks too high down to a ceiling. Pure: the input is left untouched and the result is a new array."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["limit_peak", "peak_of"]


def peak_of(samples: NDArray[np.float32]) -> float:
    """Return the largest absolute sample value of *samples*, or 0.0 for an empty array."""
    return float(np.abs(samples).max()) if samples.size else 0.0


def limit_peak(samples: NDArray[np.float32], ceiling_db: float = -1.0) -> NDArray[np.float32]:
    """Return *samples* scaled down so their peak is *ceiling_db* dBFS, or unchanged when the peak is already at or below it.

    One gain is applied to the whole clip, so the shape of the sound is kept; nothing is clipped or compressed.

    Args:
        samples: Float32 samples in any shape, where 1.0 is full scale.
        ceiling_db: The loudest peak to allow, in dBFS (0 or below).

    Returns:
        A new float32 array of the same shape.

    Raises:
        ValueError: If *ceiling_db* is above 0 dBFS.
    """
    if ceiling_db > 0:
        raise ValueError("limit_peak needs a ceiling of 0 dBFS or lower.")
    ceiling = 10 ** (ceiling_db / 20)
    peak = peak_of(samples)
    if peak <= ceiling:
        return np.array(samples, dtype=np.float32, copy=True)
    return (samples * np.float32(ceiling / peak)).astype(np.float32, copy=False)
