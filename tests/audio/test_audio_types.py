"""Tests for the audio request, voice and clip values."""

from __future__ import annotations


import numpy as np
import pytest

from zvisiongenerator.core.audio_types import AudioClip


class TestClip:
    def test_a_clip_reports_its_length(self):
        clip = AudioClip(np.zeros((2, 88200), dtype=np.float32), 44100, False)

        assert clip.seconds == 2.0
        assert clip.cut is False

    @pytest.mark.parametrize("samples", [np.zeros(10, dtype=np.float32), np.zeros((1, 2, 3), dtype=np.float32)])
    def test_samples_must_be_channels_by_frames(self, samples):
        with pytest.raises(ValueError, match=r"\(channels, frames\)"):
            AudioClip(samples, 24000)

    @pytest.mark.parametrize("dtype", [np.float64, np.int16])
    def test_samples_must_be_float32(self, dtype):
        with pytest.raises(ValueError, match="float32"):
            AudioClip(np.zeros((1, 10), dtype=dtype), 24000)

    def test_the_rate_must_be_positive(self):
        with pytest.raises(ValueError, match="sample_rate"):
            AudioClip(np.zeros((1, 10), dtype=np.float32), 0)
