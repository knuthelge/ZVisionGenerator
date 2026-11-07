"""Tests for the audio request, voice and clip values."""

from __future__ import annotations


import numpy as np

from zvisiongenerator.core.audio_types import AudioClip, AudioGenerationRequest, SpeechDelivery


class TestClip:
    def test_a_clip_reports_its_length(self):
        clip = AudioClip(np.zeros((2, 88200), dtype=np.float32), 44100, False)

        assert clip.seconds == 2.0
        assert clip.cut is False

    def test_a_clip_has_no_notes_unless_given(self):
        assert AudioClip(np.zeros((1, 10), dtype=np.float32), 10).notes == ()
        assert AudioClip(np.zeros((1, 10), dtype=np.float32), 10, notes=("Tempo was not followed.",)).notes == ("Tempo was not followed.",)


class TestRequest:
    def test_the_new_settings_start_at_auto_and_a_neutral_delivery_is_the_default_one(self):
        request = AudioGenerationRequest(preset="yue2", kind="music", prompt="pop")

        assert (request.tempo, request.key, request.delivery, request.sound_type, request.prompt_input, request.enhance_record) == (None,) * 6
        assert SpeechDelivery() == SpeechDelivery("neutral", "natural", "normal", "normal", "normal")
