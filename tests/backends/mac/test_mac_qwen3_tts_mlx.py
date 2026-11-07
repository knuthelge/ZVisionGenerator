"""Tests for the Qwen3-TTS MLX backend with fake ``mlx`` and ``mlx_audio``: the checkpoint it loads up front, seeds, the call to mlx-audio, cuts and Stop."""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests._speech_fakes import RATE, qwen_preset, tone

from zvisiongenerator.backends.audio import qwen3_tts_mlx
from zvisiongenerator.backends.audio.qwen3_tts_mlx import SAMPLE_RATE, Qwen3TtsMlxBackend
from zvisiongenerator.core.audio_backend import AudioCancelled
from zvisiongenerator.core.audio_types import AudioGenerationRequest, SpeechDelivery, VoiceRef

PRESET = qwen_preset().on_platform("mps")
CLONING_PRESET = replace(PRESET, files=PRESET.voice_files)
"""The preset as the runner hands it to the backend for a batch that follows a recording: the Base files in place of CustomVoice's."""
CUSTOM, BASE = "Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen3-TTS-12Hz-1.7B-Base"
VOICE = VoiceRef("/voices/me.wav", "Hello, this is me.")
LONG_TEXT = "The first sentence is here. " * 30  # three parts of at most 500 characters


class FakeMlxModel:
    """A stand-in for mlx-audio's Qwen3-TTS ``Model``: it records every ``generate`` call and answers with the next scripted wave.

    ``loaded`` lists every ``load`` in order, ``results`` the scripted answers (each a list of ``(wave, token_count)``, the last one
    repeating), and ``seed`` the value ``mx.random.seed`` was last given.
    """

    loaded: list[FakeMlxModel] = []
    results: list[list[tuple[np.ndarray, int]]] = []
    seed: int | None = None

    def __init__(self, folder: str) -> None:
        self.folder = folder
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any):
        self.calls.append({**kwargs, "seed": type(self).seed})
        answers = type(self).results
        for wave, tokens in answers[min(len(self.calls) - 1, len(answers) - 1)]:
            yield SimpleNamespace(audio=wave, sample_rate=RATE, token_count=tokens)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Install fake ``mlx``, ``mlx_audio`` and mac memory helpers; ``events`` notes loads, cache limits and releases in order."""
    events: list[tuple] = []
    FakeMlxModel.loaded, FakeMlxModel.results, FakeMlxModel.seed = [], [[(tone(0.5), 10)]], None

    def load(folder: str, **kwargs: Any) -> FakeMlxModel:
        assert not kwargs
        events.append(("load", Path(folder).name))
        model = FakeMlxModel(folder)
        FakeMlxModel.loaded.append(model)
        return model

    def seed(value: int) -> None:
        FakeMlxModel.seed = value

    mx = types.ModuleType("mlx.core")
    mx.random = SimpleNamespace(seed=seed)
    mx.array = lambda wave: ("mx", np.asarray(wave))
    mlx = types.ModuleType("mlx")
    mlx.core = mx
    utils = types.ModuleType("mlx_audio.tts.utils")
    utils.load = load
    for name, module in (("mlx", mlx), ("mlx.core", mx), ("mlx_audio", types.ModuleType("mlx_audio")), ("mlx_audio.tts", types.ModuleType("mlx_audio.tts")), ("mlx_audio.tts.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)

    @contextmanager
    def cache_limit(limit: int):
        events.append(("cache_limit", limit))
        try:
            yield
        finally:
            events.append(("cache_restored",))

    def load_resident(build):
        events.append(("resident",))
        return build()

    monkeypatch.setattr("zvisiongenerator.backends.mac.memory.mlx_cache_limit", cache_limit)
    monkeypatch.setattr("zvisiongenerator.backends.mac.memory.load_resident", load_resident)
    monkeypatch.setattr("zvisiongenerator.backends.mac.memory.release_memory", lambda: events.append(("release",)))
    monkeypatch.setattr(qwen3_tts_mlx, "snapshot_dir", lambda files: tmp_path / files.repo_id.rsplit("/", 1)[1])
    monkeypatch.setattr(qwen3_tts_mlx, "read_voice", lambda voice, max_seconds, rate: tone(min(2.0, max_seconds)))
    return SimpleNamespace(events=events, model=FakeMlxModel)


def _request(text: str = "Hello there.", **fields) -> AudioGenerationRequest:
    return AudioGenerationRequest(preset="qwen3_tts", kind="speech", prompt=text, **fields)


def _backend(preset=PRESET, *, cancelled=None) -> Qwen3TtsMlxBackend:
    backend = Qwen3TtsMlxBackend()
    backend.load(preset, "mps", cancelled)
    return backend


def _generate(backend: Qwen3TtsMlxBackend, request: AudioGenerationRequest, *, cancelled=lambda: False, on_progress=None):
    return backend.generate(request, on_progress or (lambda *args: None), cancelled)


class TestPreset:
    def test_the_mac_preset_runs_the_mlx_family_on_the_same_files_and_voice_files(self):
        cuda = qwen_preset()

        assert (PRESET.family, PRESET.files, PRESET.voice_files) == ("qwen3_tts_mlx", cuda.files, cuda.voice_files)


class TestLoad:
    def test_loading_reads_the_first_file_set_of_the_preset_through_load_resident(self, fake):
        _backend()

        assert [event for event in fake.events if event[0] in ("resident", "load")] == [("resident",), ("load", CUSTOM)]

    def test_a_preset_whose_files_are_the_voice_files_loads_base(self, fake):
        _backend(CLONING_PRESET)

        assert [event for event in fake.events if event[0] == "load"] == [("load", BASE)]

    def test_every_take_of_the_batch_runs_on_the_one_load(self, fake):
        backend = _backend()

        _generate(backend, _request())
        _generate(backend, _request("Another line.", seed=3))

        assert len(fake.model.loaded) == 1
        assert ("release",) not in fake.events

    def test_a_take_reports_its_parts_only_because_the_model_is_already_loaded(self, fake):
        reports = []

        _generate(_backend(), _request(), on_progress=lambda *args: reports.append(args))

        assert {report[0] for report in reports} == {"Speech"}

    def test_a_checkpoint_that_is_not_downloaded_is_a_file_not_found_error(self, fake, monkeypatch):
        def missing(files):
            raise FileNotFoundError(f"{files.repo_id} at {files.revision} is not downloaded.")

        monkeypatch.setattr(qwen3_tts_mlx, "snapshot_dir", missing)

        with pytest.raises(FileNotFoundError, match="Qwen3-TTS-12Hz-1.7B-CustomVoice"):
            _backend()

    def test_a_stop_before_the_load_loads_nothing(self, fake):
        with pytest.raises(AudioCancelled):
            _backend(cancelled=lambda: True)

        assert fake.model.loaded == []

    def test_a_stop_during_the_load_frees_what_was_loaded(self, fake):
        backend = Qwen3TtsMlxBackend()
        asked = iter([False, True])  # not before the load, then once it is done

        with pytest.raises(AudioCancelled):
            backend.load(PRESET, "mps", lambda: next(asked, True))

        assert backend._model is None

    def test_a_failed_load_leaves_no_model_behind(self, fake, monkeypatch):
        def broken(folder: str, **kwargs: Any):
            raise ValueError("Received parameters not in model")

        monkeypatch.setattr(sys.modules["mlx_audio.tts.utils"], "load", broken)
        backend = Qwen3TtsMlxBackend()

        with pytest.raises(ValueError, match="not in model"):
            backend.load(PRESET, "mps")

        assert backend._model is None

    def test_close_drops_the_model_and_a_later_load_reads_it_again(self, fake):
        backend = _backend()

        backend.close()
        backend.close()  # safe to repeat

        assert backend._model is None
        backend.load(PRESET, "mps")
        assert len(fake.model.loaded) == 2


class TestParts:
    def test_a_built_in_voice_part_calls_generate_with_the_speaker_language_and_no_splitting(self, fake):
        _generate(_backend(), _request(speaker="aiden", seed=7))

        (call,) = fake.model.loaded[0].calls
        assert call == {
            "text": "Hello there.",
            "voice": "aiden",
            "instruct": None,
            "lang_code": "auto",
            "ref_audio": None,
            "ref_text": None,
            "split_pattern": "",
            "max_tokens": 1200,
            "seed": 7,
        }

    def test_the_delivery_is_one_instruction_sentence_per_call(self, fake):
        delivery = SpeechDelivery(emotion="sad", pace="slow")

        _generate(_backend(), _request("Hello. [whisper] Quiet. [as set] Back.", delivery=delivery))

        assert [(call["text"], call["instruct"]) for call in fake.model.loaded[0].calls] == [
            ("Hello.", "Speak sadly, in a low and sorrowful voice, slowly."),
            ("Quiet.", "Whisper sadly, in a low and sorrowful voice, slowly."),
            ("Back.", "Speak sadly, in a low and sorrowful voice, slowly."),
        ]

    def test_a_neutral_delivery_sends_no_instruction(self, fake):
        _generate(_backend(), _request(delivery=SpeechDelivery()))

        assert fake.model.loaded[0].calls[0]["instruct"] is None

    def test_mlx_audio_is_not_left_to_split_a_line_break_the_text_keeps(self, fake):
        _generate(_backend(), _request("Wait…\nWhat?"))

        (call,) = fake.model.loaded[0].calls
        assert (call["text"], call["split_pattern"]) == ("Wait…\nWhat?", "")

    def test_a_recording_is_read_with_no_instruction_and_no_marks(self, fake):
        clip = _generate(_backend(CLONING_PRESET), _request("Hello. [sad] There.", voice=VOICE, delivery=SpeechDelivery(emotion="angry")))

        calls = fake.model.loaded[0].calls
        assert {call["instruct"] for call in calls} == {None}
        assert not any("[" in call["text"] for call in calls)
        assert clip.notes == ("A cloned voice reads as recorded: the Delivery marks were left out.",)

    def test_a_take_without_a_speaker_reads_in_the_default_voice(self, fake):
        _generate(_backend(), _request())

        assert fake.model.loaded[0].calls[0]["voice"] == "ryan"

    def test_each_part_is_seeded_before_it_is_read(self, fake):
        _generate(_backend(), _request(LONG_TEXT, seed=40))

        calls = fake.model.loaded[0].calls
        assert len(calls) > 1
        assert [call["seed"] for call in calls] == [40 + i for i in range(len(calls))]

    def test_the_language_is_the_presets_setting(self, fake):
        _generate(_backend(replace(PRESET, settings={**PRESET.settings, "language": "English"})), _request())

        assert fake.model.loaded[0].calls[0]["lang_code"] == "English"

    def test_a_recording_is_followed_by_every_part_with_half_a_second_of_silence_and_its_transcript(self, fake):
        _generate(_backend(CLONING_PRESET), _request(LONG_TEXT, voice=VOICE))

        calls = fake.model.loaded[0].calls
        assert len(calls) > 1
        assert {(call["voice"], call["ref_text"], call["split_pattern"]) for call in calls} == {(None, VOICE.text, "")}
        (kind, wave), *others = {id(call["ref_audio"]): call["ref_audio"] for call in calls}.values()
        assert (kind, others) == ("mx", [])  # prepared once, shared by every part
        assert wave.shape[0] == round(2.0 * RATE) + RATE // 2 and not wave[-RATE // 2 :].any() and wave.dtype == np.float32

    def test_a_recording_is_read_up_to_the_presets_longest_recording(self, fake, monkeypatch):
        asked = []
        monkeypatch.setattr(qwen3_tts_mlx, "read_voice", lambda voice, max_seconds, rate: asked.append((max_seconds, rate)) or tone(1.0))

        _generate(_backend(CLONING_PRESET), _request(voice=VOICE))

        assert asked == [(30.0, SAMPLE_RATE)]

    def test_generate_runs_inside_a_cache_limit_of_zero(self, fake):
        _generate(_backend(), _request())

        events = fake.events
        assert events.index(("cache_limit", 0)) < events.index(("cache_restored",)) and events.count(("cache_limit", 0)) >= 1
        loaded = [index for index, event in enumerate(events) if event[0] == "load"]
        assert all(index < events.index(("cache_limit", 0)) for index in loaded)

    def test_the_take_is_mono_at_24_khz_and_reports_each_part(self, fake):
        reports = []

        clip = _generate(_backend(), _request(LONG_TEXT), on_progress=lambda *args: reports.append(args))

        speech = [report for report in reports if report[0] == "Speech"]
        assert (clip.sample_rate, clip.samples.shape[0], clip.samples.dtype) == (24000, 1, np.float32)
        assert speech[0] == ("Speech", 0, 1, 1, len(speech) // 2) and speech[-1][1:3] == (1, 1)

    def test_the_results_of_a_part_are_joined_into_one_wave(self, fake):
        fake.model.results = [[(tone(0.5), 5), (tone(0.25), 3)]]
        backend = _backend()
        _generate(backend, _request())

        wave, cut = backend.read_part("Hi.", 1, speaker="ryan", reference=None, delivery=None, max_new_tokens=1200)

        assert (wave.shape[0], wave.dtype, cut) == (round(0.75 * RATE), np.float32, False)


class TestCutAndSilence:
    def test_a_part_that_fills_max_new_tokens_is_cut(self, fake):
        fake.model.results = [[(tone(5.0), 1200)]]

        assert _generate(_backend(), _request()).cut is True

    def test_a_part_that_ends_by_itself_is_not_cut(self, fake):
        fake.model.results = [[(tone(5.0), 1199)]]

        assert _generate(_backend(), _request()).cut is False

    def test_a_part_without_any_result_is_no_sound(self, fake):
        fake.model.results = [[]]

        with pytest.raises(RuntimeError, match="The model made no sound for this text. Try another seed."):
            _generate(_backend(), _request())

    def test_a_stop_between_parts_ends_the_take(self, fake):
        asked = {"parts": 0}

        def cancelled() -> bool:
            return asked["parts"] >= 1

        def on_progress(phase, step, *_):
            if phase == "Speech" and step == 1:
                asked["parts"] += 1

        with pytest.raises(AudioCancelled):
            _generate(_backend(), _request(LONG_TEXT), cancelled=cancelled, on_progress=on_progress)

        assert len(fake.model.loaded[0].calls) == 1
