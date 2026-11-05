"""Tests for the Higgs TTS 3 backend with the model, tokenizer and codec faked: loading from the cache only, parts, the voice, progress, Stop and the cut flag."""

from __future__ import annotations

import dataclasses
import gc
import subprocess
import sys
import weakref
from functools import lru_cache
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests._higgs3_fakes import CODEC_REPO, CONFIG, OFFICIAL_REPO, PRESET, FakeCodec, FakeModel, codes_of, install_fakes

from zvisiongenerator.backends.audio import higgs3 as higgs3_module
from zvisiongenerator.backends.audio import higgs3_takes
from zvisiongenerator.backends.audio.higgs3 import SAMPLE_RATE, Higgs3Backend
from zvisiongenerator.backends.audio.higgs3_takes import estimate_part_frames
from zvisiongenerator.core.audio_backend import AudioCancelled
from zvisiongenerator.core.audio_types import AudioGenerationRequest, VoiceRef
from zvisiongenerator.utils import audio_request as audio_request_module
from zvisiongenerator.utils import hf_snapshot

PART_1, PART_2, PART_3 = ("a" * 299 + ".", "b" * 299 + ".", "c" * 299 + ".")
THREE_PARTS = f"{PART_1} {PART_2} {PART_3}"


def _tone(seconds: float, amplitude: float = 0.4) -> torch.Tensor:
    return torch.from_numpy((amplitude * np.cos(2 * np.pi * 220 * np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE)).astype(np.float32))


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """Return a loaded backend over fakes; ``rig.model.takes`` and ``rig.codec.waves`` script the takes, ``rig.calls`` records each part's generation."""
    events: list = []
    model, codec = FakeModel(), FakeCodec(events, [])
    fakes = install_fakes(monkeypatch, tmp_path, model, codec)
    calls: list[SimpleNamespace] = []
    real_generate = higgs3_module.generate_speech_codes

    def spy(model_, tokenizer, ids, text, **kwargs):
        calls.append(SimpleNamespace(text=text, ids=ids, seed=torch.initial_seed(), **kwargs))
        return real_generate(model_, tokenizer, ids, text, **kwargs)

    monkeypatch.setattr(higgs3_module, "generate_speech_codes", spy)
    monkeypatch.setattr(higgs3_takes, "probe_track_seconds", lambda path: 10.0)
    monkeypatch.setattr(audio_request_module, "probe_track_seconds", lambda path: 10.0)
    backend = Higgs3Backend()
    backend.load(PRESET, "cuda")
    return SimpleNamespace(backend=backend, model=model, codec=codec, events=events, calls=calls, fakes=fakes)


def _request(prompt: str = PART_1, *, voice: VoiceRef | None = None, seed: int = 7) -> AudioGenerationRequest:
    return AudioGenerationRequest(preset="higgs3", kind="speech", prompt=prompt, voice=voice, seed=seed)


def _generate(rig, request, *, cancelled=lambda: False):
    progress: list[tuple] = []
    clip = rig.backend.generate(request, lambda *args: progress.append(args), cancelled)
    return clip, progress


def _with_cap(rig, max_new_tokens: int) -> None:
    rig.backend._limits = dataclasses.replace(rig.backend._limits, max_new_tokens=max_new_tokens)


class TestLoad:
    def test_the_model_is_built_from_the_official_snapshots_config_and_weights_on_the_device(self, rig):
        rig.fakes.read_config.assert_called_once_with(rig.fakes.official / "config.json")
        rig.fakes.load_model.assert_called_once_with(CONFIG, rig.fakes.official / "model.safetensors", "cuda", None)

    def test_the_tokenizer_comes_from_the_official_snapshot(self, rig):
        rig.fakes.tokenizer_load.assert_called_once_with(rig.fakes.official)

    def test_the_codec_loads_from_the_pinned_codec_snapshot_in_float32_without_remote_code(self, rig):
        rig.fakes.codec_load.assert_called_once_with(str(rig.fakes.codec_dir), dtype=torch.float32)
        assert ("codec.to", "cpu") in rig.events

    def test_loading_never_downloads(self, monkeypatch, tmp_path):
        import huggingface_hub

        def refuse(*_args, **_kwargs):
            raise AssertionError("load() must not download")

        events: list = []
        install_fakes(monkeypatch, tmp_path, FakeModel(), FakeCodec(events, []))
        monkeypatch.setattr(hf_snapshot, "snapshot_download", refuse)
        monkeypatch.setattr(hf_snapshot, "snapshot_with", refuse)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)
        backend = Higgs3Backend()

        backend.load(PRESET, "cuda")

        assert backend._model is not None

    @pytest.mark.parametrize("repo_id", [OFFICIAL_REPO, CODEC_REPO])
    def test_a_repository_missing_from_the_cache_is_refused_by_name_before_any_load(self, rig, repo_id):
        rig.fakes.snapshots[repo_id] = []
        backend = Higgs3Backend()

        with pytest.raises(FileNotFoundError, match=f"{repo_id} at [0-9a-f]{{40}} is not downloaded"):
            backend.load(PRESET, "cuda")

        rig.fakes.load_model.reset_mock()
        rig.fakes.codec_load.reset_mock()
        assert backend._model is None
        rig.fakes.load_model.assert_not_called()
        rig.fakes.codec_load.assert_not_called()

    def test_the_stop_check_reaches_the_weight_load(self, rig):
        def check() -> bool:
            return False

        Higgs3Backend().load(PRESET, "cuda", cancelled=check)

        assert rig.fakes.load_model.call_args.args[3] is check

    def test_a_stop_during_the_weight_load_leaves_the_backend_unloaded_and_never_loads_the_codec(self, rig):
        rig.fakes.load_model.side_effect = AudioCancelled
        rig.fakes.codec_load.reset_mock()
        backend = Higgs3Backend()

        with pytest.raises(AudioCancelled):
            backend.load(PRESET, "cuda", cancelled=lambda: True)

        assert backend._model is None
        rig.fakes.codec_load.assert_not_called()

    def test_a_stop_before_anything_loads_reads_neither_the_tokenizer_nor_the_model(self, rig):
        rig.fakes.tokenizer_load.reset_mock()
        rig.fakes.load_model.reset_mock()
        backend = Higgs3Backend()

        with pytest.raises(AudioCancelled):
            backend.load(PRESET, "cuda", cancelled=lambda: True)

        rig.fakes.tokenizer_load.assert_not_called()
        rig.fakes.load_model.assert_not_called()
        assert backend._model is None

    def test_a_stop_after_the_tokenizer_is_read_builds_no_model(self, rig):
        stopped: list[bool] = []
        rig.fakes.tokenizer_load.side_effect = lambda folder: stopped.append(True) or MagicMock()
        rig.fakes.load_model.reset_mock()
        backend = Higgs3Backend()

        with pytest.raises(AudioCancelled):
            backend.load(PRESET, "cuda", cancelled=lambda: bool(stopped))

        rig.fakes.load_model.assert_not_called()

    def test_a_stop_after_the_model_is_read_skips_the_codec_and_frees_the_model(self, rig):
        stopped: list[bool] = []
        rig.fakes.load_model.side_effect = lambda *args: stopped.append(True) or FakeModel()
        rig.fakes.codec_load.reset_mock()
        backend = Higgs3Backend()

        with pytest.raises(AudioCancelled):
            backend.load(PRESET, "cuda", cancelled=lambda: bool(stopped))

        assert backend._model is None
        rig.fakes.codec_load.assert_not_called()

    def test_a_stop_while_the_codec_is_built_is_seen_when_it_returns(self, rig):
        stopped: list[bool] = []
        original = rig.fakes.codec_load.return_value
        rig.fakes.codec_load.side_effect = lambda *args, **kwargs: stopped.append(True) or original
        backend = Higgs3Backend()

        with pytest.raises(AudioCancelled):
            backend.load(PRESET, "cuda", cancelled=lambda: bool(stopped))

        assert backend._model is None and backend._codec is None

    def test_a_failed_load_leaves_the_backend_unloaded(self, rig):
        rig.fakes.codec_load.side_effect = OSError("no codec")
        backend = Higgs3Backend()

        with pytest.raises(OSError, match="no codec"):
            backend.load(PRESET, "cuda")

        assert backend._model is None

    def test_the_codec_decodes_on_its_own_device_and_drops_the_marker_codes(self, rig):
        rig.codec.waves.append(_tone(0.1))

        wave = rig.backend._decode(torch.tensor([[1, 1025], [1024, 5], [7, 8]]))

        (batch,) = rig.codec.decoded
        assert batch.shape == (1, 2, 3)
        assert batch.dtype == torch.long
        assert batch[0].tolist() == [[1, 0, 7], [0, 5, 8]]
        assert wave.ndim == 1
        assert wave.dtype == np.float32

    def test_the_codec_runs_on_the_cpu_on_cuda_and_on_mps_on_a_mac_by_default(self, rig):
        assert ("codec.to", "cpu") in rig.events

        Higgs3Backend().load(PRESET, "mps")

        assert ("codec.to", "mps") in rig.events
        assert rig.fakes.load_model.call_args.args[2] == "mps"

    def test_a_preset_can_put_the_codec_on_another_device(self, rig):
        settings = {**PRESET.settings, "codec_device": "cuda"}
        Higgs3Backend().load(dataclasses.replace(PRESET, settings=settings), "cuda")

        assert ("codec.to", "cuda") in rig.events


class TestParts:
    def test_one_part_is_generated_with_the_presets_settings(self, rig):
        rig.model.takes[:] = [100]
        rig.codec.waves.append(_tone(1.0))

        clip, _ = _generate(rig, _request(seed=7))

        (call,) = rig.calls
        assert (call.text, call.reference_codes, call.reference_text) == (PART_1, None, None)
        assert (call.max_new_tokens, call.temperature, call.top_p, call.seed) == (1500, 0.7, 0.95, 7)
        assert clip.samples.shape == (1, SAMPLE_RATE)
        assert clip.sample_rate == 24000
        assert clip.samples.dtype == np.float32
        assert not clip.cut

    def test_a_loud_take_keeps_its_peaks_above_full_scale_for_the_runner_to_scale(self, rig):
        rig.model.takes[:] = [100]
        rig.codec.waves.append(_tone(1.0, amplitude=1.6))

        clip, _ = _generate(rig, _request())

        assert float(np.abs(clip.samples).max()) > 1.0

    def test_the_codes_of_a_part_reach_the_codec_without_the_frame_that_ended_it(self, rig):
        rig.model.takes[:] = [6]  # 6 head calls: 5 kept frames, 5 - (N - 1) = 4 of them sound
        rig.codec.waves.append(_tone(1.0))

        _generate(rig, _request())

        (batch,) = rig.codec.decoded
        assert batch.shape == (1, 2, 4)

    def test_three_parts_are_seeded_one_apart_and_joined_into_one_clip(self, rig):
        rig.model.takes[:] = [10, 10, 10]
        rig.codec.waves.extend([_tone(0.5), _tone(0.5), _tone(0.5)])

        clip, _ = _generate(rig, _request(THREE_PARTS, seed=10))

        assert [call.text for call in rig.calls] == [PART_1, PART_2, PART_3]
        assert [call.seed for call in rig.calls] == [10, 11, 12]
        assert clip.samples.shape == (1, 3 * 12000 + 2 * 7200)

    def test_with_any_the_first_part_is_the_reference_of_the_rest(self, rig):
        rig.codec.waves.extend([_tone(0.5), _tone(0.5), _tone(0.5)])

        _generate(rig, _request(THREE_PARTS))

        first, second, third = rig.calls
        assert (first.reference_codes, first.reference_text) == (None, None)
        for call in (second, third):
            assert torch.equal(call.reference_codes, codes_of(0))
            assert call.reference_codes.dtype == torch.long
            assert call.reference_text == PART_1
        (encoded,) = rig.codec.encoded
        assert encoded.shape == (1, 1, SAMPLE_RATE)  # part 1's 0.5 s take, padded to one second

    def test_the_first_part_is_encoded_as_it_was_generated(self, rig):
        rig.codec.waves.extend([_tone(2.0), _tone(0.5)])

        _generate(rig, _request(f"{PART_1} {PART_2}"))

        (encoded,) = rig.codec.encoded
        assert encoded.shape == (1, 1, 2 * SAMPLE_RATE)
        np.testing.assert_allclose(encoded.reshape(-1).numpy(), _tone(2.0).numpy())

    def test_a_single_part_makes_no_reference(self, rig):
        rig.codec.waves.append(_tone(0.5))

        _generate(rig, _request())

        assert rig.codec.encoded == []

    def test_with_a_recording_every_part_follows_it_and_nothing_is_chained(self, rig, monkeypatch):
        read = MagicMock(return_value=np.full((2, 10 * 48000), 0.1, dtype=np.float32))
        monkeypatch.setattr(higgs3_takes, "read_track", read)
        rig.codec.waves.extend([_tone(2.0), _tone(0.5), _tone(0.5)])

        _generate(rig, _request(THREE_PARTS, voice=VoiceRef("me.wav", "Hello, this is me.")))

        (recording,) = rig.codec.encoded  # the recording once; part 1's take is not encoded
        assert recording.shape == (1, 1, 10 * SAMPLE_RATE)  # read at 48 kHz, polyphase-resampled to 24 kHz
        assert read.call_args.args[0].path == "me.wav"
        assert read.call_args.args[1:] == (48000, 10.0)
        assert len(rig.calls) == 3
        for call in rig.calls:
            assert torch.equal(call.reference_codes, codes_of(0))  # the recording's codes
            assert call.reference_text == "Hello, this is me."

    def test_with_any_a_reference_longer_than_the_voice_limit_is_cut_to_it_with_its_words(self, rig):
        long_take = _tone(40.0)  # the preset allows a voice 30 s long
        words = " ".join(f"w{n}" for n in range(100))
        part_1 = words + "." * (300 - len(words))
        rig.codec.waves.extend([long_take, _tone(0.5)])

        _generate(rig, _request(f"{part_1} {PART_2}"))

        (encoded,) = rig.codec.encoded
        assert encoded.shape == (1, 1, 30 * SAMPLE_RATE)
        np.testing.assert_allclose(encoded.reshape(-1).numpy(), long_take[: 30 * SAMPLE_RATE].numpy())
        reference_text = rig.calls[1].reference_text
        assert reference_text == " ".join(part_1.split()[:75])  # 30 of 40 seconds keeps three quarters of the words

    def test_with_any_a_reference_within_the_voice_limit_is_kept_whole(self, rig):
        rig.codec.waves.extend([_tone(29.0), _tone(0.5)])

        _generate(rig, _request(f"{PART_1} {PART_2}"))

        (encoded,) = rig.codec.encoded
        assert encoded.shape == (1, 1, 29 * SAMPLE_RATE)
        assert rig.calls[1].reference_text == PART_1

    def test_a_stereo_recording_is_averaged_to_mono_before_it_is_encoded(self, rig, monkeypatch):
        stereo = np.stack([np.full(10 * 48000, 0.2, dtype=np.float32), np.full(10 * 48000, -0.2, dtype=np.float32)])  # opposite channels cancel
        monkeypatch.setattr(higgs3_takes, "read_track", lambda *args: stereo)
        rig.codec.waves.append(_tone(0.5))

        _generate(rig, _request(voice=VoiceRef("me.wav", "Hello, this is me.")))

        (recording,) = rig.codec.encoded
        assert float(recording.abs().max()) < 1e-6

    def test_the_codec_gets_its_input_and_output_on_its_own_device(self, rig):
        rig.backend._codec_device = "meta"  # a device other than the model's, which no test machine lacks
        rig.codec.waves.append(_tone(0.1))

        rig.backend.encode(np.zeros(SAMPLE_RATE, dtype=np.float32))
        rig.backend._decode(torch.zeros(3, 2, dtype=torch.long))

        assert rig.codec.encoded[0].device.type == "meta"
        assert rig.codec.decoded[0].device.type == "meta"

    def test_a_recording_with_one_part_is_encoded_once_and_not_chained(self, rig, monkeypatch):
        monkeypatch.setattr(higgs3_takes, "read_track", lambda *args: np.full((2, 10 * 48000), 0.1, dtype=np.float32))
        rig.codec.waves.append(_tone(0.5))

        _generate(rig, _request(voice=VoiceRef("me.wav", "Hello, this is me.")))

        assert len(rig.codec.encoded) == 1
        assert rig.calls[0].reference_text == "Hello, this is me."

    def test_a_silent_first_part_stops_an_any_take(self, rig):
        _with_cap(rig, 1)  # fewer iterations than codebooks: the part has no sound

        with pytest.raises(RuntimeError, match="first part"):
            _generate(rig, _request(f"{PART_1} {PART_2}"))

        assert len(rig.calls) == 1

    def test_a_take_with_no_sound_at_all_is_an_error(self, rig):
        _with_cap(rig, 1)

        with pytest.raises(RuntimeError, match=r"^Higgs TTS 3 made no sound for this text\. Try another seed\.$"):
            _generate(rig, _request())


class TestProgress:
    def test_the_first_frame_and_every_tenth_are_reported_with_the_part(self, rig):
        rig.model.takes[:] = [26, 3]
        rig.codec.waves.extend([_tone(0.5), _tone(0.5)])

        _, progress = _generate(rig, _request(f"{PART_1} {PART_2}"))

        speech = [args for args in progress if args[0] == "Speech tokens"]
        assert [(step, part) for _, step, _, part, _ in speech] == [(1, 1), (10, 1), (20, 1), (26, 1), (1, 2), (3, 2)]
        assert {n_parts for *_, n_parts in speech} == {2}

    def test_a_part_reports_the_frames_its_text_should_need_not_the_cap(self, rig):
        rig.model.takes[:] = [26]
        rig.codec.waves.append(_tone(0.5))

        _, progress = _generate(rig, _request(PART_1))

        expected = estimate_part_frames(PART_1, 1500)
        speech = [args for args in progress if args[0] == "Speech tokens"]
        assert 0 < expected < 1500
        assert [total for _, _, total, *_ in speech[:-1]] == [expected] * (len(speech) - 1)

    def test_a_finished_part_ends_with_its_own_frame_count_as_the_total(self, rig):
        rig.model.takes[:] = [26, 20]
        rig.codec.waves.extend([_tone(0.5), _tone(0.5)])

        _, progress = _generate(rig, _request(f"{PART_1} {PART_2}"))

        speech = [args for args in progress if args[0] == "Speech tokens"]
        assert [args for args in speech if args[3] == 1][-1][1:3] == (26, 26)
        assert [args for args in speech if args[3] == 2][-1][1:3] == (20, 20)

    def test_a_part_that_runs_past_its_estimate_keeps_the_bar_just_short_of_full(self, rig):
        _with_cap(rig, 600)
        rig.model.takes[:] = [500]
        rig.codec.waves.append(_tone(0.5))

        _, progress = _generate(rig, _request("short text"))

        running = [(step, total) for phase, step, total, *_ in progress if phase == "Speech tokens"][:-1]
        assert len(running) > 3
        assert all(step < total <= 600 for step, total in running)

    def test_each_part_reports_its_decode_after_its_frames(self, rig):
        rig.model.takes[:] = [3, 3]
        rig.codec.waves.extend([_tone(0.5), _tone(0.5)])

        _, progress = _generate(rig, _request(f"{PART_1} {PART_2}"))

        assert [args for args in progress if args[0] == "Decode"] == [("Decode", 0, 1, 1, 2), ("Decode", 1, 1, 1, 2), ("Decode", 0, 1, 2, 2), ("Decode", 1, 1, 2, 2)]
        assert [args[0] for args in progress[:4]] == ["Speech tokens"] * 2 + ["Decode"] * 2


class TestEstimate:
    def test_it_grows_with_the_length_of_the_text(self):
        short, long = (estimate_part_frames("a" * length, 1500) for length in (100, 400))

        assert 0 < short < long < 1500

    def test_a_five_hundred_character_part_is_in_the_range_measured_on_real_takes(self):
        assert 650 <= estimate_part_frames("a" * 500, 1500) <= 800

    def test_it_is_never_below_one_frame_or_above_the_cap(self):
        assert estimate_part_frames("", 1500) == 1
        assert estimate_part_frames("a" * 5000, 1500) == 1500


class TestStop:
    def test_stopping_between_parts_ends_the_take_before_the_next_part(self, rig):
        rig.model.takes[:] = [10, 10, 10]
        rig.codec.waves.extend([_tone(0.5), _tone(0.5), _tone(0.5)])

        with pytest.raises(AudioCancelled):
            _generate(rig, _request(THREE_PARTS), cancelled=lambda: len(rig.calls) >= 1)

        assert len(rig.calls) == 1

    def test_stopping_before_the_first_part_generates_nothing(self, rig):
        with pytest.raises(AudioCancelled):
            _generate(rig, _request(), cancelled=lambda: True)

        assert rig.calls == []

    def test_stopping_is_checked_on_every_frame_not_only_the_reported_ones(self, rig):
        rig.model.takes[:] = [1000]
        steps: list[int] = []
        checks: list[int] = []
        progress = lambda phase, step, *_: steps.append(step)  # noqa: E731

        def cancelled() -> bool:
            checks.append(len(checks))
            return len(checks) >= 14  # check 1 is before the part; frames 1 to 12 pass; the 13th frame is stopped

        with pytest.raises(AudioCancelled):
            rig.backend.generate(_request(), progress, cancelled)

        assert steps == [1, 10]  # frames 11 and 12 were counted but not reported, and the stop at frame 13 did not wait for frame 20
        assert rig.model.head_calls == 13
        assert rig.codec.decoded == []

    def test_the_backend_makes_the_next_take_after_a_stop(self, rig):
        rig.model.takes[:] = [1000, 20]
        steps: list[int] = []

        with pytest.raises(AudioCancelled):
            rig.backend.generate(_request(), lambda phase, step, *_: steps.append(step), lambda: len(steps) >= 3)
        rig.codec.waves.append(_tone(0.5))
        clip, _ = _generate(rig, _request())

        assert clip.samples.shape[1] > 0


class TestCut:
    def test_a_take_that_ends_by_itself_is_not_cut(self, rig):
        _with_cap(rig, 40)
        rig.model.takes[:] = [39]
        rig.codec.waves.append(_tone(0.5))

        clip, _ = _generate(rig, _request())

        assert not clip.cut

    def test_a_part_that_reaches_the_cap_cuts_the_take(self, rig):
        _with_cap(rig, 40)
        rig.model.takes[:] = [10, 5000, 10]
        rig.codec.waves.extend([_tone(0.5), _tone(0.5), _tone(0.5)])

        clip, _ = _generate(rig, _request(THREE_PARTS))

        assert clip.cut

    def test_a_cut_part_is_still_decoded_and_heard(self, rig):
        _with_cap(rig, 40)
        rig.model.takes[:] = [5000]
        rig.codec.waves.append(_tone(0.5))

        clip, _ = _generate(rig, _request())

        (batch,) = rig.codec.decoded
        assert batch.shape == (1, 2, 39)  # every kept frame: 40 - (N - 1)
        assert clip.samples.shape[1] > 0


class TestClose:
    def test_close_frees_the_model_and_the_codec(self, rig):
        rig.backend.close()

        assert rig.backend._model is None
        assert rig.backend._codec is None

    def test_nothing_keeps_the_model_or_the_codec_alive_after_close(self, monkeypatch, tmp_path):
        # After close() a weak reference to each must be dead, so the accelerator memory they own can be freed.
        events: list = []
        made: dict = {}

        def make_model(*_args, **_kwargs):
            made["model"] = weakref.ref(model := FakeModel([3]))
            return model

        def make_codec(*_args, **_kwargs):
            made["codec"] = weakref.ref(codec := FakeCodec(events, [_tone(1.0)]))
            return codec

        fakes = install_fakes(monkeypatch, tmp_path, None, None)
        fakes.load_model.side_effect = make_model
        fakes.codec_load.side_effect = make_codec
        backend = Higgs3Backend()
        backend.load(PRESET, "cuda")
        backend.generate(_request(), lambda *args: None, lambda: False)
        assert made["model"]() is not None and made["codec"]() is not None

        backend.close()
        gc.collect()

        assert made["model"]() is None
        assert made["codec"]() is None
        assert backend._tokenizer is None and backend._ids is None

    def test_close_frees_a_codec_held_by_the_conv_layer_cache(self, monkeypatch):
        # transformers' codec caches its conv layers in an lru_cache keyed by the codec itself.
        class CachingCodec:
            @lru_cache
            def _get_conv1d_layers(self, module):
                return ()

        codec_module = ModuleType(higgs3_module._CODEC_MODULE)
        codec_module.HiggsAudioV2TokenizerPreTrainedModel = CachingCodec
        monkeypatch.setitem(sys.modules, higgs3_module._CODEC_MODULE, codec_module)
        codec = CachingCodec()
        codec._get_conv1d_layers(None)
        held = weakref.ref(codec)
        del codec

        Higgs3Backend().close()
        gc.collect()

        assert held() is None

    def test_close_warns_instead_of_raising_when_the_cache_is_gone(self, monkeypatch):
        codec_module = ModuleType(higgs3_module._CODEC_MODULE)
        codec_module.HiggsAudioV2TokenizerPreTrainedModel = type("Codec", (), {"_get_conv1d_layers": lambda self, module: ()})
        monkeypatch.setitem(sys.modules, higgs3_module._CODEC_MODULE, codec_module)

        with pytest.warns(UserWarning, match="cache was not found"):
            Higgs3Backend().close()

    def test_close_without_transformers_loaded_imports_nothing(self, monkeypatch):
        monkeypatch.delitem(sys.modules, higgs3_module._CODEC_MODULE, raising=False)

        Higgs3Backend().close()

        assert higgs3_module._CODEC_MODULE not in sys.modules


class TestImports:
    def test_importing_the_backend_loads_neither_torch_nor_transformers_nor_the_network(self):
        """The network module needs torch to define its classes, so only a load imports it."""
        code = (
            "import sys\n"
            "import zvisiongenerator.backends.audio.higgs3, zvisiongenerator.backends.audio.higgs3_generate\n"
            "import zvisiongenerator.backends.audio.higgs3_codes, zvisiongenerator.backends.audio.higgs3_prompt, zvisiongenerator.backends.audio.higgs3_config\n"
            "heavy = {'torch', 'transformers', 'zvisiongenerator.backends.audio.higgs3_model'}\n"
            "raise SystemExit(bool(heavy & {name.split('.')[0] if name != 'zvisiongenerator.backends.audio.higgs3_model' else name for name in sys.modules}))"
        )

        assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
