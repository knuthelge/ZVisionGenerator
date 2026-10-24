"""Tests for zvisiongenerator.utils.audio_resample."""

from __future__ import annotations

import numpy as np

from zvisiongenerator.utils.audio_resample import resample_audio


def _stereo(samples: int, dtype: str = "float32") -> np.ndarray:
    t = np.arange(samples) / 32000
    return np.stack([np.sin(2 * np.pi * 440 * t), np.cos(2 * np.pi * 220 * t)]).astype(dtype)


def test_output_length_scales_with_rate_ratio() -> None:
    out = resample_audio(_stereo(32000), 32000, 48000)

    assert out.shape == (2, 48000)


def test_equal_rates_return_input_unchanged() -> None:
    audio = _stereo(1000)

    assert resample_audio(audio, 48000, 48000) is audio


def test_output_is_float32_and_keeps_channel_count() -> None:
    out = resample_audio(_stereo(3200, dtype="float64"), 32000, 48000)

    assert out.dtype == np.float32
    assert out.shape[0] == 2


def test_mono_channel_shape_is_preserved() -> None:
    out = resample_audio(_stereo(3200)[:1], 32000, 16000)

    assert out.shape == (1, 1600)
