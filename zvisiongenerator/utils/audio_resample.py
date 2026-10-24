"""Polyphase resampling of multi-channel audio."""

from __future__ import annotations

import math

import numpy as np

__all__ = ["resample_audio"]


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Return ``(channels, samples)`` audio resampled from *source_rate* to *target_rate* (polyphase).

    Args:
        audio: Audio shaped ``(channels, samples)``.
        source_rate: Sample rate of *audio* in Hz.
        target_rate: Sample rate to resample to in Hz.

    Returns:
        *audio* itself when the rates are equal, otherwise a new float32 array.
    """
    if source_rate == target_rate:
        return audio
    from scipy.signal import resample_poly

    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // divisor, source_rate // divisor, axis=1).astype("float32")
