"""Tests for the Qwen3-TTS backend with a fake ``faster_qwen3_tts``: the checkpoint it loads up front, the shim, seeds, cuts and Stop."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from tests._speech_fakes import RATE, install_fake_faster_qwen, qwen_preset, tone

from zvisiongenerator.backends.audio.qwen3_tts import SAMPLE_RATE, Qwen3TtsBackend
from zvisiongenerator.core.audio_backend import AudioCancelled
from zvisiongenerator.core.audio_types import AudioGenerationRequest, SpeechDelivery, VoiceRef

PRESET = qwen_preset()
CLONING_PRESET = replace(PRESET, files=PRESET.voice_files)
"""The preset as the runner hands it to the backend for a batch that follows a recording: the Base files in place of CustomVoice's."""
CUSTOM, BASE = "Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen3-TTS-12Hz-1.7B-Base"
VOICE = VoiceRef("/voices/me.wav", "Hello, this is me.")
LONG_TEXT = "The first sentence is here. " * 30  # three parts of at most 500 characters


def _request(text: str = "Hello there.", **fields) -> AudioGenerationRequest:
    return AudioGenerationRequest(preset="qwen3_tts", kind="speech", prompt=text, **fields)


def _backend(preset=PRESET, *, cancelled=None) -> Qwen3TtsBackend:
    backend = Qwen3TtsBackend()
    backend.load(preset, "cuda", cancelled)
    return backend


def _generate(backend: Qwen3TtsBackend, request: AudioGenerationRequest, *, cancelled=lambda: False, on_progress=None):
    return backend.generate(request, on_progress or (lambda *args: None), cancelled)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    return install_fake_faster_qwen(monkeypatch, tmp_path)


class TestLoad:
    def test_loading_reads_the_first_file_set_of_the_preset_with_its_graphs_captured(self, fake):
        _backend()

        (load,) = fake.model.loaded
        assert load["folder"].endswith(CUSTOM)
        assert (load["device"], load["dtype"], load["local_files_only"]) == ("cuda", torch.bfloat16, True)
        assert load["instance"].warmups == 1

    def test_a_preset_whose_files_are_the_voice_files_loads_base(self, fake):
        _backend(CLONING_PRESET)

        (load,) = fake.model.loaded
        assert load["folder"].endswith(BASE)
        assert load["instance"].warmups == 1

    def test_the_shim_is_applied_before_the_load(self, fake):
        _backend()

        assert [event for event in fake.events if event[0] in ("shim", "load")] == [("shim",), ("load", CUSTOM)]

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

        monkeypatch.setattr("zvisiongenerator.backends.audio.qwen3_tts.snapshot_dir", missing)

        with pytest.raises(FileNotFoundError, match="Qwen3-TTS-12Hz-1.7B-CustomVoice"):
            _backend()

    def test_a_stop_before_the_load_loads_nothing(self, fake):
        with pytest.raises(AudioCancelled):
            _backend(cancelled=lambda: True)

        assert fake.model.loaded == []

    def test_a_stop_while_the_checkpoint_builds_ends_the_load_and_keeps_nothing(self, fake, monkeypatch):
        stop = {"asked": False}

        def building(cls, folder, **kwargs):
            stop["asked"] = True  # the Stop arrives while the library creates its layers
            torch.nn.Linear(1, 1)
            raise AssertionError("the build should have been stopped at the first layer")

        monkeypatch.setattr(fake.model, "from_pretrained", classmethod(building))
        backend = Qwen3TtsBackend()

        with pytest.raises(AudioCancelled):
            backend.load(PRESET, "cuda", lambda: stop["asked"])

        assert backend._model is None

    def test_a_failed_warmup_leaves_no_model_behind(self, fake):
        fake.model.fail_warmup = True
        backend = Qwen3TtsBackend()

        with pytest.raises(RuntimeError, match="graph capture failed"):
            backend.load(PRESET, "cuda")

        assert backend._model is None

    def test_close_drops_the_model_and_a_later_load_reads_it_again(self, fake):
        backend = _backend()

        backend.close()
        backend.close()  # safe to repeat

        assert backend._model is None
        backend.load(PRESET, "cuda")
        assert len(fake.model.loaded) == 2


class TestParts:
    def test_a_built_in_voice_part_calls_custom_voice_with_the_speaker_language_and_no_instruction(self, fake):
        _generate(_backend(), _request(speaker="aiden", seed=7))

        (call,) = fake.model.loaded[0]["instance"].calls
        assert call == {"kind": "custom_voice", "text": "Hello there.", "speaker": "aiden", "language": "auto", "instruct": None, "max_new_tokens": 1200, "seed": 7}

    def test_the_delivery_is_one_instruction_sentence_per_call(self, fake):
        delivery = SpeechDelivery(emotion="sad", pace="slow")

        _generate(_backend(), _request("Hello. [whisper] Quiet. [as set] Back.", delivery=delivery))

        calls = fake.model.loaded[0]["instance"].calls
        assert [(call["text"], call["instruct"]) for call in calls] == [
            ("Hello.", "Speak sadly, in a low and sorrowful voice, slowly."),
            ("Quiet.", "Whisper sadly, in a low and sorrowful voice, slowly."),
            ("Back.", "Speak sadly, in a low and sorrowful voice, slowly."),
        ]

    def test_a_neutral_delivery_sends_no_instruction(self, fake):
        _generate(_backend(), _request(delivery=SpeechDelivery()))

        assert fake.model.loaded[0]["instance"].calls[0]["instruct"] is None

    def test_a_mark_changes_the_instruction_of_its_own_segment_only(self, fake):
        _generate(_backend(), _request("One. [shout] Two. [as set] Three."))

        assert [call["instruct"] for call in fake.model.loaded[0]["instance"].calls] == [None, "Shout loudly.", None]

    def test_a_recording_is_read_with_no_instruction_and_no_marks(self, fake):
        clip = _generate(_backend(CLONING_PRESET), _request("Hello. [sad] There.", voice=VOICE, delivery=SpeechDelivery(emotion="angry")))

        calls = fake.model.loaded[0]["instance"].calls
        assert {call["kind"] for call in calls} == {"voice_clone"}
        assert all("instruct" not in call and "[" not in call["text"] for call in calls)
        assert clip.notes == ("A cloned voice reads as recorded: the Delivery marks were left out.",)

    def test_a_take_without_a_speaker_reads_in_the_default_voice(self, fake):
        _generate(_backend(), _request())

        assert fake.model.loaded[0]["instance"].calls[0]["speaker"] == "ryan"

    def test_each_part_is_seeded_before_it_is_read(self, fake):
        _generate(_backend(), _request(LONG_TEXT, seed=40))

        calls = fake.model.loaded[0]["instance"].calls
        assert len(calls) > 1
        assert [call["seed"] for call in calls] == [40 + i for i in range(len(calls))]

    def test_the_language_is_the_presets_setting(self, fake):
        preset = qwen_preset()
        backend = Qwen3TtsBackend()
        backend.load(replace(preset, settings={**preset.settings, "language": "English"}), "cuda")

        _generate(backend, _request())

        assert fake.model.loaded[0]["instance"].calls[0]["language"] == "English"

    def test_a_recording_is_prepared_once_with_half_a_second_of_silence_and_followed_by_every_part(self, fake):
        _generate(_backend(CLONING_PRESET), _request(LONG_TEXT, voice=VOICE))

        instance = fake.model.loaded[0]["instance"]
        (prompt,) = instance.prompts
        wave, rate = prompt["ref_audio"]
        assert (rate, prompt["ref_text"]) == (SAMPLE_RATE, VOICE.text)
        assert wave.shape[0] == round(2.0 * RATE) + RATE // 2 and not wave[-RATE // 2 :].any()
        assert len(instance.calls) > 1
        assert {call["kind"] for call in instance.calls} == {"voice_clone"}
        assert {tuple(call["voice_clone_prompt"]) for call in instance.calls} == {(f"prompt:{VOICE.text}",)}

    def test_a_recording_is_read_up_to_the_presets_longest_recording(self, fake, monkeypatch):
        asked = []
        monkeypatch.setattr("zvisiongenerator.backends.audio.qwen3_tts.read_voice", lambda voice, max_seconds, rate: asked.append((max_seconds, rate)) or tone(1.0))

        _generate(_backend(CLONING_PRESET), _request(voice=VOICE))

        assert asked == [(30.0, SAMPLE_RATE)]

    def test_the_take_is_mono_at_24_khz_and_reports_each_part(self, fake):
        reports = []

        clip = _generate(_backend(), _request(LONG_TEXT), on_progress=lambda *args: reports.append(args))

        speech = [report for report in reports if report[0] == "Speech"]
        assert (clip.sample_rate, clip.samples.shape[0], clip.samples.dtype) == (24000, 1, np.float32)
        assert speech[0] == ("Speech", 0, 1, 1, len(speech) // 2) and speech[-1][1:3] == (1, 1)


class TestCutAndSilence:
    def test_a_part_that_fills_max_new_tokens_frames_is_cut(self, fake):
        full = np.full(int(1200 / 12.5 * RATE), 0.1, dtype=np.float32)
        fake.model.waves = [full]

        assert _generate(_backend(), _request()).cut is True

    def test_a_part_one_frame_short_of_the_limit_still_counts_as_cut(self, fake):
        fake.model.waves = [np.full(int(2047 / 12.5 * RATE), 0.1, dtype=np.float32)]

        assert _generate(_backend(), _request()).cut is True

    def test_a_part_that_ends_by_itself_is_not_cut(self, fake):
        fake.model.waves = [tone(5.0)]

        assert _generate(_backend(), _request()).cut is False

    def test_the_single_zero_sample_of_a_part_without_codes_is_no_sound(self, fake):
        fake.model.waves = [np.zeros(1, dtype=np.float32)]

        with pytest.raises(RuntimeError, match="The model made no sound for this text. Try another seed."):
            _generate(_backend(), _request())
