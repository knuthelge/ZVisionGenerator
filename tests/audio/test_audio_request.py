"""Tests for the checks of an audio request: the length, the speech text and voice limits, and the request a form makes."""

from __future__ import annotations

import pytest

from tests._audio_presets import audio_config, sa3_two_kind_preset
from tests.audio._helpers import preset

from zvisiongenerator.core.audio_types import AudioGenerationRequest, SpeechDelivery, VoiceRef
from zvisiongenerator.utils import audio_request as audio_request_module
from zvisiongenerator.utils.audio_preset import parse_audio_presets
from zvisiongenerator.utils.audio_request import auto_seconds, check_speech_request, resolve_audio_seconds
from zvisiongenerator.utils.audio_track import TRACK_UNREADABLE
from zvisiongenerator.web.audio_job import AudioFields, audio_request_from_fields

LIMITS = preset("qwen3_tts").limits


def _request(prompt: str = "Hello there.", *, voice: VoiceRef | None = None) -> AudioGenerationRequest:
    return AudioGenerationRequest(preset="qwen3_tts", kind="speech", prompt=prompt, voice=voice, seed=7)


class TestCheckSpeechRequest:
    def test_text_over_the_limit_is_refused(self):
        with pytest.raises(ValueError, match="10,000"):
            check_speech_request(_request("a" * 10_001), LIMITS)

    def test_text_at_the_limit_is_accepted(self):
        check_speech_request(_request("a " * 5000), LIMITS)

    def test_an_unknown_mark_is_refused_by_name_before_the_model_loads(self):
        with pytest.raises(ValueError, match=r"^\[sadd\] isn't a Delivery mark\. Use one of the Delivery marks, or remove the brackets\.$"):
            check_speech_request(_request("Hello. [sadd] There."), LIMITS)

    def test_an_unknown_mark_is_refused_for_a_recording_too(self):
        with pytest.raises(ValueError, match=r"\[whispr\] isn't a Delivery mark"):
            check_speech_request(_request("Hi [whispr]", voice=VoiceRef("me.wav", "hi")), LIMITS, check_voice_file=False)

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("Wait [pause] now", "[pause] isn't a Delivery mark. For a pause, type … (the Insert menu adds one)."),
            ("Wait [Pauses] now", "[Pauses] isn't a Delivery mark. For a pause, type … (the Insert menu adds one)."),
            ("Ha [laugh] ha", "[laugh] isn't a Delivery mark. For a laugh, write haha (the Insert menu adds one)."),
            ("Ha [laughs] ha", "[laughs] isn't a Delivery mark. For a laugh, write haha (the Insert menu adds one)."),
            ("Ha [laughing] ha", "[laughing] isn't a Delivery mark. For a laugh, write haha (the Insert menu adds one)."),
        ],
    )
    def test_a_pause_or_laugh_mark_is_refused_with_what_to_write_instead(self, text, message):
        with pytest.raises(ValueError) as error:
            check_speech_request(_request(text), LIMITS)

        assert str(error.value) == message

    def test_known_marks_and_other_brackets_are_accepted(self):
        check_speech_request(_request("Hi. [sad] There [1]. [as set] [Whisper] Ok."), LIMITS)

    def test_marks_do_not_count_against_the_text_limit(self):
        check_speech_request(_request("[sad]" + "a " * 5000), LIMITS)
        with pytest.raises(ValueError, match="10,000"):
            check_speech_request(_request("[sad]" + "a" * 10_001), LIMITS)

    def test_text_of_only_marks_is_empty_text(self):
        with pytest.raises(ValueError, match="Write the text"):
            check_speech_request(_request("[sad] [whisper]"), LIMITS)

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


class TestResolveAudioSeconds:
    @pytest.mark.parametrize(("sound_type", "expected"), [(None, 12.0), ("auto", 12.0), ("ambience", 12.0), ("effect", 5.0), ("one-shot", 2.0)])
    def test_a_sound_on_auto_has_the_length_of_its_type(self, sound_type, expected):
        assert resolve_audio_seconds(None, preset("sa3"), "sound", sound_type, name="seconds") == expected

    def test_a_chosen_length_wins_over_the_type(self):
        assert resolve_audio_seconds(7.5, preset("sa3"), "sound", "one-shot", name="seconds") == 7.5

    def test_music_from_a_model_that_sets_its_length_is_thirty_seconds_on_auto(self):
        stable_audio = _two_kind_preset()

        assert resolve_audio_seconds(None, stable_audio, "music", None, name="seconds") == 30.0
        assert resolve_audio_seconds(90, stable_audio, "music", None, name="seconds") == 90.0

    def test_a_model_that_sets_its_own_length_has_none(self):
        assert resolve_audio_seconds(None, preset("yue2"), "music", None, name="seconds") is None
        assert resolve_audio_seconds(None, preset("qwen3_tts"), "speech", None, name="seconds") is None

    @pytest.mark.parametrize("seconds", [0.5, 381.0, float("nan"), float("inf")])
    def test_a_length_outside_the_models_range_is_refused_by_the_name_it_was_given(self, seconds):
        with pytest.raises(ValueError, match=r"^--seconds must be between 1 and 380"):
            resolve_audio_seconds(seconds, preset("sa3"), "sound", "auto", name="--seconds")

    def test_the_auto_length_of_a_kind(self):
        assert (auto_seconds("sound"), auto_seconds("sound", "effect"), auto_seconds("music")) == (12.0, 5.0, 30.0)


def _two_kind_preset():
    return next(item for item in parse_audio_presets(audio_config(sa3=sa3_two_kind_preset())))


class TestRequestFromFields:
    def test_a_song_reads_its_tempo_and_key(self):
        request = audio_request_from_fields(preset("yue2"), "music", AudioFields(prompt="synth-pop", tempo="92", key="F#m"))

        assert (request.tempo, request.key) == (92, "F#m")
        assert (request.delivery, request.sound_type, request.seconds) == (None, None, None)

    def test_tempo_and_key_blank_are_auto(self):
        request = audio_request_from_fields(preset("yue2"), "music", AudioFields(prompt="synth-pop", tempo="", key=""))

        assert (request.tempo, request.key) == (None, None)

    def test_music_of_a_model_that_sets_its_length_has_the_resolved_length(self):
        stable_audio = _two_kind_preset()

        assert audio_request_from_fields(stable_audio, "music", AudioFields(prompt="pad")).seconds == 30.0
        assert audio_request_from_fields(stable_audio, "music", AudioFields(prompt="pad", seconds=45.0)).seconds == 45.0

    def test_a_sound_resolves_its_type_and_length(self):
        auto = audio_request_from_fields(preset("sa3"), "sound", AudioFields(prompt="rain"))
        effect = audio_request_from_fields(preset("sa3"), "sound", AudioFields(prompt="door", sound_type="effect"))
        chosen = audio_request_from_fields(preset("sa3"), "sound", AudioFields(prompt="door", sound_type="effect", seconds=8.0))

        assert [(r.sound_type, r.seconds) for r in (auto, effect, chosen)] == [("auto", 12.0), ("effect", 5.0), ("effect", 8.0)]

    def test_speech_in_a_built_in_voice_has_a_delivery_and_a_recording_has_none(self, monkeypatch):
        monkeypatch.setattr(audio_request_module, "probe_track_seconds", lambda path: 10.0)
        qwen = preset("qwen3_tts")

        built_in = audio_request_from_fields(qwen, "speech", AudioFields(prompt="Hi", emotion="sad", pace="slow"))
        default = audio_request_from_fields(qwen, "speech", AudioFields(prompt="Hi"))
        recording = audio_request_from_fields(qwen, "speech", AudioFields(prompt="Hi", voice_path="me.wav", voice_text="Hello", voice_consent=True, emotion="sad"))

        assert built_in.delivery == SpeechDelivery(emotion="sad", pace="slow")
        assert default.delivery == SpeechDelivery()
        assert recording.delivery is None and recording.speaker is None and recording.voice == VoiceRef("me.wav", "Hello")

    def test_a_setting_of_another_kind_is_ignored_even_when_wrong(self):
        sound = audio_request_from_fields(preset("sa3"), "sound", AudioFields(prompt="rain", tempo="fast", key="H", emotion="grumpy"))
        song = audio_request_from_fields(preset("yue2"), "music", AudioFields(prompt="pop", sound_type="x", emotion="grumpy"))

        assert (sound.tempo, sound.key, sound.delivery) == (None, None, None)
        assert (song.sound_type, song.delivery) == (None, None)

    @pytest.mark.parametrize(
        ("model", "kind", "fields", "message"),
        [
            ("yue2", "music", {"tempo": "300"}, "Tempo must be 40 to 220 BPM."),
            ("yue2", "music", {"tempo": "slow"}, "Tempo must be 40 to 220 BPM."),
            ("yue2", "music", {"key": "H"}, "Key must be one of:"),
            ("qwen3_tts", "speech", {"emotion": "grumpy"}, "Emotion must be one of:"),
            ("qwen3_tts", "speech", {"pace": "warp"}, "Pace must be one of:"),
            ("sa3", "sound", {"sound_type": "song"}, "Sound type must be one of:"),
            ("sa3", "sound", {"seconds": 400.0}, "seconds must be between 1 and 380"),
        ],
    )
    def test_a_bad_value_of_the_kinds_own_setting_is_refused_in_user_wording(self, model, kind, fields, message):
        with pytest.raises(ValueError, match=message):
            audio_request_from_fields(preset(model), kind, AudioFields(prompt="x", **fields))

    def test_an_enhanced_text_keeps_the_users_own_text_even_when_empty(self):
        record = {"settings": {}, "model": "fast", "tier": "fast", "format": "text", "task": "style"}

        request = audio_request_from_fields(preset("yue2"), "music", AudioFields(prompt="dreamy lo-fi", prompt_input="", enhance_record=record))

        assert (request.prompt, request.prompt_input, request.enhance_record) == ("dreamy lo-fi", "", record)

    def test_an_own_text_without_a_record_is_not_kept(self):
        request = audio_request_from_fields(preset("yue2"), "music", AudioFields(prompt="dreamy lo-fi", prompt_input="lo-fi"))

        assert (request.prompt_input, request.enhance_record) == (None, None)
