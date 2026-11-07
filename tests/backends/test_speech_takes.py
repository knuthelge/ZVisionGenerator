"""Tests for the model-neutral speech take: parts and seeds, the voice every part follows, progress, Stop, the join and the refusals."""

from __future__ import annotations

import numpy as np
import pytest

from tests._speech_fakes import RATE, FakeSpeechEngine, qwen_preset, tone

from zvisiongenerator.backends.audio.speech_takes import MARKS_LEFT_OUT_NOTE, SPEECH_PHASE, read_speech, read_voice
from zvisiongenerator.core.audio_backend import AudioCancelled
from zvisiongenerator.core.audio_types import AudioGenerationRequest, SpeechDelivery, VoiceRef
from zvisiongenerator.utils.audio_preset import AudioLimits
from zvisiongenerator.utils.speech_text import speech_parts

PRESET = qwen_preset()
SETTINGS = PRESET.settings_for("cuda")
LIMITS = AudioLimits(max_new_tokens=300, part_chars=40, max_text_chars=1000, min_voice_seconds=3, max_voice_seconds=30)
LONG_TEXT = "The first sentence is here. The second sentence follows it. A third one ends the text."


def _long_text_parts() -> tuple[str, ...]:
    """Return the texts LONG_TEXT is read in at this preset's part size."""
    return tuple(part.text for part in speech_parts(LONG_TEXT, None, part_chars=40, max_text_chars=1000))


def _request(text: str = "Hello there.", **fields) -> AudioGenerationRequest:
    return AudioGenerationRequest(preset="qwen3_tts", kind="speech", prompt=text, **fields)


def _read(engine, request, *, on_progress=None, cancelled=lambda: False):
    return read_speech(engine, request, PRESET, LIMITS, SETTINGS, on_progress or (lambda *args: None), cancelled)


class TestParts:
    def test_a_long_text_is_read_in_parts_and_part_i_is_seeded_seed_plus_i(self):
        engine = FakeSpeechEngine()
        n = len(_long_text_parts())
        assert n > 1

        _read(engine, _request(LONG_TEXT, seed=100))

        assert [part["seed"] for part in engine.parts] == [100 + i for i in range(n)]
        assert [part["text"] for part in engine.parts] == list(_long_text_parts())
        assert all(part["max_new_tokens"] == 300 for part in engine.parts)

    def test_the_take_is_the_parts_joined_at_the_engines_rate_as_one_channel(self):
        engine = FakeSpeechEngine([tone(0.5)])
        n = len(_long_text_parts())

        clip = _read(engine, _request(LONG_TEXT))

        assert clip.sample_rate == RATE and clip.samples.shape[0] == 1
        assert clip.seconds > 0.5 * n * 0.9  # the parts are all in it, less the edge trim and plus the pauses

    def test_the_take_is_cut_when_any_part_is(self):
        assert _read(FakeSpeechEngine(cuts=(1,)), _request(LONG_TEXT)).cut is True
        assert _read(FakeSpeechEngine(), _request(LONG_TEXT)).cut is False


class TestVoice:
    def test_a_built_in_voice_take_reads_every_part_in_the_requested_speaker(self):
        engine = FakeSpeechEngine()

        _read(engine, _request(LONG_TEXT, speaker="aiden"))

        assert {part["speaker"] for part in engine.parts} == {"aiden"}
        assert {part["reference"] for part in engine.parts} == {None}
        assert engine.prepared == []

    def test_a_take_that_names_no_speaker_reads_in_the_presets_default_voice(self):
        engine = FakeSpeechEngine()

        _read(engine, _request())

        assert engine.parts[0]["speaker"] == "ryan"

    def test_a_recording_is_prepared_once_and_followed_by_every_part(self):
        engine = FakeSpeechEngine()
        voice = VoiceRef("/voices/me.wav", "Hello, this is me.")

        _read(engine, _request(LONG_TEXT, voice=voice, speaker="aiden"))

        assert engine.prepared == [voice]
        assert {part["reference"] for part in engine.parts} == {("prompt", "Hello, this is me.")}
        assert {part["speaker"] for part in engine.parts} == {None}  # the recording decides the voice, whatever the speaker field says

    def test_the_recording_is_prepared_before_the_first_part(self):
        events: list = []
        engine = FakeSpeechEngine(events=events)

        _read(engine, _request(voice=VoiceRef("/voices/me.wav", "Hi")))

        assert events == ["prepare", ("part", 0)]

    def test_a_speaker_the_preset_does_not_have_is_refused_with_the_ones_it_has(self):
        with pytest.raises(ValueError, match=r"no voice 'bob'; use one of ryan, aiden"):
            _read(FakeSpeechEngine(), _request(speaker="bob"))

    def test_a_preset_without_built_in_voices_refuses_a_take_without_a_recording(self):
        from dataclasses import replace

        bare = replace(PRESET, voices=())

        with pytest.raises(ValueError, match="has no built-in voice"):
            read_speech(FakeSpeechEngine(), _request(), bare, LIMITS, SETTINGS, lambda *args: None, lambda: False)


class TestDelivery:
    """Each part is read in the Delivery of the segment of marked text it belongs to; a recording is read as recorded."""

    def test_every_part_of_unmarked_text_gets_the_requests_delivery(self):
        engine = FakeSpeechEngine()
        delivery = SpeechDelivery(emotion="sad", pace="slow")

        _read(engine, _request(LONG_TEXT, delivery=delivery))

        assert len(engine.parts) > 1
        assert {part["delivery"] for part in engine.parts} == {delivery}

    def test_a_request_without_a_delivery_reads_at_the_neutral_one(self):
        engine = FakeSpeechEngine()

        _read(engine, _request("Hello."))

        assert engine.parts[0]["delivery"] == SpeechDelivery()

    def test_marks_give_each_segment_its_own_delivery_and_are_not_read(self):
        engine = FakeSpeechEngine()
        delivery = SpeechDelivery(pace="slow")

        _read(engine, _request("Hello there. [sad] Oh no. [as set] Fine now.", delivery=delivery, seed=10))

        assert [(part["text"], part["delivery"], part["seed"]) for part in engine.parts] == [
            ("Hello there.", delivery, 10),
            ("Oh no.", SpeechDelivery(emotion="sad", pace="slow"), 11),
            ("Fine now.", delivery, 12),
        ]

    def test_a_line_break_in_the_text_reaches_the_engine(self):
        engine = FakeSpeechEngine()

        _read(engine, _request("Wait…\nWhat?"))

        assert [part["text"] for part in engine.parts] == ["Wait…\nWhat?"]

    def test_progress_counts_the_parts_the_marks_make(self):
        reports: list[tuple] = []

        _read(FakeSpeechEngine(), _request("One. [sad] Two. [angry] Three."), on_progress=lambda *args: reports.append(args))

        assert reports == [(SPEECH_PHASE, step, 1, part, 3) for part in (1, 2, 3) for step in (0, 1)]

    def test_a_recording_ignores_the_delivery_and_the_marks(self):
        engine = FakeSpeechEngine()
        delivery = SpeechDelivery(emotion="sad")

        clip = _read(engine, _request("Hello. [angry] There.", voice=VoiceRef("/voices/me.wav", "Hi"), delivery=delivery))

        assert [(part["text"], part["delivery"]) for part in engine.parts] == [("Hello. There.", None)]
        assert clip.notes == (MARKS_LEFT_OUT_NOTE,)
        assert MARKS_LEFT_OUT_NOTE == "A cloned voice reads as recorded: the Delivery marks were left out."

    def test_a_recording_without_marks_gets_no_note(self):
        clip = _read(FakeSpeechEngine(), _request("Hello [1].", voice=VoiceRef("/voices/me.wav", "Hi")))

        assert clip.notes == ()

    def test_a_built_in_voice_with_marks_gets_no_note(self):
        assert _read(FakeSpeechEngine(), _request("Hello. [sad] There.")).notes == ()

    def test_a_text_of_only_marks_makes_no_sound(self):
        with pytest.raises(RuntimeError, match="no sound"):
            _read(FakeSpeechEngine(), _request("[sad]"))


class TestProgress:
    def test_each_part_reports_before_and_after_it(self):
        reports: list[tuple] = []
        engine = FakeSpeechEngine()
        n = len(_long_text_parts())

        _read(engine, _request(LONG_TEXT), on_progress=lambda *args: reports.append(args))

        assert reports == [(SPEECH_PHASE, step, 1, part, n) for part in range(1, n + 1) for step in (0, 1)]
        assert SPEECH_PHASE == "Speech"

    def test_progress_for_a_part_comes_around_the_part_itself(self):
        events: list = []
        engine = FakeSpeechEngine(events=events)

        _read(engine, _request(LONG_TEXT), on_progress=lambda phase, step, total, part, n: events.append(("progress", step, part)))

        assert events[:4] == [("progress", 0, 1), ("part", 0), ("progress", 1, 1), ("progress", 0, 2)]


class TestStop:
    def test_stop_is_checked_before_each_part_and_before_the_join(self):
        checks: list[int] = []
        n = len(_long_text_parts())

        _read(FakeSpeechEngine(), _request(LONG_TEXT), cancelled=lambda: checks.append(1) or False)

        assert len(checks) == n + 1

    def test_stop_before_the_second_part_ends_the_take_after_the_first(self):
        engine = FakeSpeechEngine()
        answers = iter([False, True])

        with pytest.raises(AudioCancelled):
            _read(engine, _request(LONG_TEXT), cancelled=lambda: next(answers))

        assert len(engine.parts) == 1

    def test_stop_during_the_last_part_ends_the_take_before_the_join(self, monkeypatch):
        n = len(_long_text_parts())
        answers = iter([False] * n + [True])
        joined = []
        monkeypatch.setattr("zvisiongenerator.backends.audio.speech_takes.join_speech_parts", lambda *args: joined.append(1))

        with pytest.raises(AudioCancelled):
            _read(FakeSpeechEngine(), _request(LONG_TEXT), cancelled=lambda: next(answers))

        assert joined == []


class TestNoSound:
    def test_a_take_where_every_part_is_silent_says_so(self):
        engine = FakeSpeechEngine([np.zeros(0, dtype=np.float32)])

        with pytest.raises(RuntimeError, match=r"^The model made no sound for this text\. Try another seed\.$"):
            _read(engine, _request(LONG_TEXT))

    def test_one_silent_part_among_sounding_ones_is_joined_over(self):
        engine = FakeSpeechEngine([tone(0.5), np.zeros(0, dtype=np.float32), tone(0.5)])

        clip = _read(engine, _request(LONG_TEXT))

        assert clip.seconds > 0.5


class TestReadVoice:
    def test_the_recording_is_read_at_twice_the_rate_up_to_the_limit_and_resampled(self, monkeypatch):
        calls = {}
        stereo = np.ones((2, 4 * 48000), dtype=np.float32)

        def read_track(track, rate, seconds):
            calls["read"] = (track.path, rate, seconds)
            return stereo

        monkeypatch.setattr("zvisiongenerator.backends.audio.speech_takes.probe_track_seconds", lambda path: 50.0)
        monkeypatch.setattr("zvisiongenerator.backends.audio.speech_takes.read_track", read_track)

        wave = read_voice(VoiceRef("/voices/me.wav", "x"), 30.0, RATE)

        assert calls["read"] == ("/voices/me.wav", 2 * RATE, 30.0)
        assert wave.ndim == 1 and wave.shape[0] == 4 * RATE
