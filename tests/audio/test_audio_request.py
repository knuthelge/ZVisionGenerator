"""Tests for the checks of an audio request: the speech text and voice limits."""

from __future__ import annotations

import pytest

from tests.audio._helpers import preset

from zvisiongenerator.core.audio_types import AudioGenerationRequest, VoiceRef
from zvisiongenerator.utils import audio_request as audio_request_module
from zvisiongenerator.utils.audio_request import check_speech_request
from zvisiongenerator.utils.audio_track import TRACK_UNREADABLE

LIMITS = preset("higgs3").limits


def _request(prompt: str = "Hello there.", *, voice: VoiceRef | None = None) -> AudioGenerationRequest:
    return AudioGenerationRequest(preset="higgs3", kind="speech", prompt=prompt, voice=voice, seed=7)


class TestCheckSpeechRequest:
    def test_text_over_the_limit_is_refused_with_its_message(self):
        with pytest.raises(ValueError, match=r"^Shorten the text: it's over 10,000 characters, about 10 minutes of speech\.$"):
            check_speech_request(_request("a" * 10_001), LIMITS)

    def test_text_at_the_limit_is_accepted(self):
        check_speech_request(_request("a " * 5000), LIMITS)

    def test_empty_text_is_refused(self):
        with pytest.raises(ValueError, match="Write the text"):
            check_speech_request(_request("  "), LIMITS)

    @pytest.mark.parametrize("seconds", [2.9, 30.1])
    def test_a_voice_outside_3_to_30_seconds_is_refused(self, monkeypatch, seconds):
        monkeypatch.setattr(audio_request_module, "probe_track_seconds", lambda path: seconds)

        with pytest.raises(ValueError, match=r"between 3 and 30 seconds"):
            check_speech_request(_request(voice=VoiceRef("me.wav", "hi")), LIMITS)

    @pytest.mark.parametrize("seconds", [3.0, 12.0, 30.0])
    def test_a_voice_inside_the_range_is_accepted(self, monkeypatch, seconds):
        monkeypatch.setattr(audio_request_module, "probe_track_seconds", lambda path: seconds)

        check_speech_request(_request(voice=VoiceRef("me.wav", "hi")), LIMITS)

    def test_an_unreadable_voice_is_refused_with_the_tracks_message(self, monkeypatch):
        def unreadable(path):
            raise ValueError(TRACK_UNREADABLE)

        monkeypatch.setattr(audio_request_module, "probe_track_seconds", unreadable)

        with pytest.raises(ValueError, match=TRACK_UNREADABLE):
            check_speech_request(_request(voice=VoiceRef("gone.wav", "hi")), LIMITS)

    def test_the_voice_file_is_not_read_when_asked_not_to(self, monkeypatch):
        def unreadable(path):
            raise AssertionError("must not be read")

        monkeypatch.setattr(audio_request_module, "probe_track_seconds", unreadable)

        check_speech_request(_request(voice=VoiceRef("gone.wav", "hi")), LIMITS, check_voice_file=False)
