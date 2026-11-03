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
    """Return *samples* scaled by one gain so their peak is *ceiling_db* dBFS; nothing is clipped or compressed.

    The runner calls it only for a clip that peaks above full scale.
    """
    ceiling = 10 ** (ceiling_db / 20)
    peak = peak_of(samples)
    return (samples * np.float32(ceiling / peak)).astype(np.float32, copy=False)
