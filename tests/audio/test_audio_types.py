"""Tests for the audio request, voice and clip values."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from zvisiongenerator.core.audio_types import AudioClip, AudioGenerationRequest, VoiceRef


class TestRequest:
    def test_a_request_holds_what_the_take_needs(self):
        voice = VoiceRef("voice.wav", "Hello there.")

        request = AudioGenerationRequest(preset="higgs3", kind="speech", prompt="Read this.", lyrics="", seconds=None, voice=voice, seed=7)

        assert (request.preset, request.kind, request.prompt, request.seconds, request.seed) == ("higgs3", "speech", "Read this.", None, 7)
        assert request.voice == voice

    def test_the_optional_fields_default_to_none_or_empty(self):
        request = AudioGenerationRequest(preset="sa3", kind="sound", prompt="Rain")

        assert (request.lyrics, request.seconds, request.voice) == ("", None, None)

    def test_a_request_is_frozen(self):
        request = AudioGenerationRequest(preset="sa3", kind="sound", prompt="Rain")

        with pytest.raises(dataclasses.FrozenInstanceError):
            request.prompt = "Wind"  # type: ignore[misc]

    def test_a_voice_may_come_without_a_transcript(self):
        assert VoiceRef("voice.wav").text is None

    def test_a_voice_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            VoiceRef("voice.wav", "x").path = "other.wav"  # type: ignore[misc]


class TestClip:
    def test_a_clip_reports_its_length(self):
        clip = AudioClip(np.zeros((2, 88200), dtype=np.float32), 44100, False)

        assert clip.seconds == 2.0
        assert clip.cut is False

    def test_cut_defaults_to_false_and_is_kept(self):
        samples = np.zeros((1, 24000), dtype=np.float32)

        assert AudioClip(samples, 24000).cut is False
        assert AudioClip(samples, 24000, cut=True).cut is True

    def test_a_clip_is_frozen(self):
        clip = AudioClip(np.zeros((1, 10), dtype=np.float32), 24000)

        with pytest.raises(dataclasses.FrozenInstanceError):
            clip.cut = True  # type: ignore[misc]

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
