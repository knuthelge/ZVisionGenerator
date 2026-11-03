"""Tests for the audio request, voice and clip values."""

from __future__ import annotations


import numpy as np

from zvisiongenerator.core.audio_types import AudioClip


class TestClip:
    def test_a_clip_reports_its_length(self):
        clip = AudioClip(np.zeros((2, 88200), dtype=np.float32), 44100, False)

        assert clip.seconds == 2.0
        assert clip.cut is False
