"""Tests for the audio runner with the backend mocked: order of steps, saving, level, out of memory, closing and the job controls."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests._audio_presets import audio_config
from tests._higgs3_fakes import FakeCodec, FakeModel, install_fakes
from tests._media import REAL
from tests.audio._helpers import clip, decode, fake_backend, patch_runner, preset, sine

from zvisiongenerator import audio_runner
from zvisiongenerator.audio_runner import run_audio_batch
from zvisiongenerator.backends.audio.higgs3 import SAMPLE_RATE, Higgs3Backend
from zvisiongenerator.core.audio_backend import AudioCancelled
from zvisiongenerator.core.audio_types import AudioGenerationRequest, VoiceRef
from zvisiongenerator.downloads import DETACHED, STOPPED
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.provenance import read_config

CONFIG = audio_config()
YUE2, HIGGS3, SA3 = preset("yue2"), preset("higgs3"), preset("sa3")


def _music(**fields) -> AudioGenerationRequest:
    return AudioGenerationRequest(preset="yue2", kind="music", prompt="synth-pop", lyrics="[verse]\nla la", **fields)


def _speech(text: str = "Hello there.", **fields) -> AudioGenerationRequest:
    return AudioGenerationRequest(preset="higgs3", kind="speech", prompt=text, **fields)


def _sound(**fields) -> AudioGenerationRequest:
    return AudioGenerationRequest(preset="sa3", kind="sound", prompt="rain", seconds=2.0, **fields)


def _run(request, model, tmp_path, **overrides):
    options = {"runs": 1, "seed": 5, "output_dir": str(tmp_path / "out")}
    options.update(overrides)
    return run_audio_batch(request, model, CONFIG, **options)


class TestOrder:
    def test_the_enhancer_is_released_then_the_files_are_ensured_then_the_model_loads(self, monkeypatch, tmp_path):
        order: list[str] = []
        backend = fake_backend(clip(), events=order)
        env = patch_runner(monkeypatch, backend)
        env.session.release_resident_enhancer.side_effect = lambda: order.append("release")
        env.ensure.side_effect = lambda *a, **k: order.append("ensure") or "ready"

        _run(_music(), YUE2, tmp_path)

        assert order == ["release", "ensure", "load", "generate", "close"]

    def test_the_download_plan_is_the_presets_files_on_this_platform(self, monkeypatch, tmp_path):
        env = patch_runner(monkeypatch, fake_backend(clip()))

        _run(_music(), YUE2, tmp_path)

        targets = env.ensure.call_args.args[0]
        assert [target.repo_id for target in targets] == ["m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"]

    def test_the_backend_of_the_platform_entrys_family_loads_once_for_all_takes(self, monkeypatch, tmp_path):
        backend = fake_backend(clip())
        patch_runner(monkeypatch, backend)

        _run(_music(), YUE2, tmp_path, runs=3)

        audio_runner.get_audio_backend.assert_called_once_with("yue2")
        backend.load.assert_called_once_with(YUE2, YUE2.platforms["cuda"], "cuda")
        assert backend.generate.call_count == 3 and backend.close.call_count == 1

    def test_a_speech_request_is_checked_before_anything_loads(self, monkeypatch, tmp_path):
        order: list[str] = []
        env = patch_runner(monkeypatch, fake_backend(clip(), events=order))
        monkeypatch.setattr(audio_runner, "check_speech_request", lambda request, limits: order.append("check"))
        env.session.release_resident_enhancer.side_effect = lambda: order.append("release")

        _run(_speech(), HIGGS3, tmp_path)

        assert order[:2] == ["check", "release"] and "load" in order

    def test_a_speech_text_over_the_limit_fails_before_anything_loads(self, monkeypatch, tmp_path):
        backend = fake_backend(clip())
        env = patch_runner(monkeypatch, backend)

        with pytest.raises(ValueError, match="10,000 characters"):
            _run(_speech("x" * 10001), HIGGS3, tmp_path)

        env.session.release_resident_enhancer.assert_not_called()
        env.ensure.assert_not_called()
        audio_runner.get_audio_backend.assert_not_called()

    def test_a_voice_outside_its_length_fails_before_anything_loads(self, monkeypatch, tmp_path):
        env = patch_runner(monkeypatch, fake_backend(clip()))
        from zvisiongenerator.utils import audio_request

        monkeypatch.setattr(audio_request, "probe_track_seconds", lambda path: 45.0)

        with pytest.raises(ValueError, match="between 3 and 30 seconds"):
            _run(_speech(voice=VoiceRef("me.wav", "hi")), HIGGS3, tmp_path)

        env.ensure.assert_not_called()

    def test_higgs_downloads_its_two_repositories_and_loads_with_no_step_between(self, monkeypatch, tmp_path):
        order: list[str] = []
        env = patch_runner(monkeypatch, fake_backend(clip(), events=order))
        env.ensure.side_effect = lambda *a, **k: order.append("download") or "ready"

        _run(_speech(), HIGGS3, tmp_path)

        assert order[:2] == ["download", "load"]
        assert [target.repo_id for target in env.ensure.call_args.args[0]] == ["bosonai/higgs-tts-3-4b", "bosonai/higgs-audio-v2-tokenizer"]

    def test_a_model_that_does_not_run_on_this_platform_is_refused_before_anything_happens(self, monkeypatch, tmp_path):
        env = patch_runner(monkeypatch, fake_backend(clip()))
        monkeypatch.setattr(audio_runner, "audio_platform_key", lambda: "mps")
        only_cuda = SimpleNamespace(**{**YUE2.__dict__, "platforms": {"cuda": YUE2.platforms["cuda"]}})

        with pytest.raises(ValueError, match="does not run on 'mps'"):
            _run(_music(), only_cuda, tmp_path)

        env.session.release_resident_enhancer.assert_not_called()

    def test_a_request_for_another_model_is_refused(self, monkeypatch, tmp_path):
        patch_runner(monkeypatch, fake_backend(clip()))

        with pytest.raises(ValueError, match="not music model 'yue2'"):
            _run(_sound(), YUE2, tmp_path)

    def test_a_stopped_download_ends_the_job_without_loading(self, monkeypatch, tmp_path):
        events: list[dict] = []
        backend = fake_backend(clip())
        patch_runner(monkeypatch, backend, ensure_outcome=STOPPED)

        saved = _run(_music(), YUE2, tmp_path, progress_callback=events.append)

        assert saved == [] and backend.load.call_count == 0
        assert events[-1]["type"] == "batch_cancelled" and events[-1]["download"] == STOPPED

    def test_a_detached_download_ends_the_job_too(self, monkeypatch, tmp_path):
        events: list[dict] = []
        patch_runner(monkeypatch, fake_backend(clip()), ensure_outcome=DETACHED)

        _run(_music(), YUE2, tmp_path, progress_callback=events.append)

        assert events[-1]["download"] == DETACHED


class TestSaving:
    @REAL
    def test_each_take_is_a_flac_with_its_settings_and_its_own_seed(self, monkeypatch, tmp_path):
        patch_runner(monkeypatch, fake_backend(clip(0.5)))

        saved = _run(_music(), YUE2, tmp_path, runs=2, seed=None)

        assert len(saved) == 2 and all(path.suffix == ".flac" and path.name.startswith("music_") for path in saved)
        configs = [read_config(path) for path in saved]
        assert [config["kind"] for config in configs] == ["music", "music"]
        assert configs[0]["prompt"] == "synth-pop" and configs[0]["lyrics"] == "[verse]\nla la" and configs[0]["model"] == "yue2"
        assert configs[0]["seed"] != configs[1]["seed"]

    @REAL
    def test_a_fixed_seed_is_used_for_every_take_and_passed_to_the_backend(self, monkeypatch, tmp_path):
        backend = fake_backend(clip(0.2))
        patch_runner(monkeypatch, backend)

        saved = _run(_music(), YUE2, tmp_path, runs=2, seed=42)

        assert [call.args[0].seed for call in backend.generate.call_args_list] == [42, 42]
        assert read_config(saved[0])["seed"] == 42
        assert saved[0] != saved[1]

    @REAL
    def test_the_output_folder_is_created_and_a_cut_take_says_so(self, monkeypatch, tmp_path):
        patch_runner(monkeypatch, fake_backend(clip(0.2, cut=True)))

        saved = _run(_music(), YUE2, tmp_path, output_dir=str(tmp_path / "deep" / "er"))

        assert saved[0].parent == tmp_path / "deep" / "er" and read_config(saved[0])["generation"]["cut"] is True

    @REAL
    def test_a_speech_take_records_the_full_text_and_the_number_of_parts(self, monkeypatch, tmp_path):
        patch_runner(monkeypatch, fake_backend(clip(0.2)))
        sentence = "This sentence has a fair number of characters in it, give or take. "
        text = sentence * 20  # about 1,340 characters: three parts of up to 500

        monkeypatch.setattr(audio_runner, "check_speech_request", lambda request, limits: None)
        saved = _run(_speech(text, voice=VoiceRef("me.wav", "Hello, this is me.")), HIGGS3, tmp_path)

        config = read_config(saved[0])
        assert config["prompt"] == text and config["generation"]["parts"] == 3
        assert config["voice_path"].endswith("me.wav") and config["voice_text"] == "Hello, this is me."
        assert not any("consent" in key for key in config)

    @REAL
    def test_a_sound_records_the_length_asked_for(self, monkeypatch, tmp_path):
        patch_runner(monkeypatch, fake_backend(clip(0.5, rate=44100, channels=2)))

        saved = _run(_sound(), SA3, tmp_path)

        config = read_config(saved[0])
        _, rate = decode(saved[0], channels=2)
        assert config["seconds"] == 2.0 and config["generation"]["duration"] == 0.5 and rate == 44100


class TestLevel:
    @REAL
    def test_a_take_before_the_clamp_that_peaks_over_full_scale_is_brought_to_minus_1_dbfs(self, monkeypatch, tmp_path):
        loud = sine(0.3, rate=44100, amplitude=10 ** (6.3 / 20))
        patch_runner(monkeypatch, fake_backend(clip_from(loud, 44100, preclamp=True)))

        saved = _run(_sound(), SA3, tmp_path)

        decoded, _ = decode(saved[0])
        assert 20 * np.log10(np.abs(decoded).max()) == pytest.approx(-1.0, abs=0.01)

    @REAL
    def test_a_take_before_the_clamp_that_stays_within_full_scale_is_saved_as_it_is(self, monkeypatch, tmp_path):
        samples = sine(0.3, rate=44100, amplitude=0.95)
        patch_runner(monkeypatch, fake_backend(clip_from(samples, 44100, preclamp=True)))

        decoded, _ = decode(_run(_sound(), SA3, tmp_path)[0])

        assert np.abs(decoded).max() == pytest.approx(0.95, abs=0.001)

    @REAL
    def test_a_clip_that_is_not_before_the_clamp_is_not_scaled(self, monkeypatch, tmp_path):
        samples = sine(0.3, rate=44100, amplitude=1.5)
        patch_runner(monkeypatch, fake_backend(clip_from(samples, 44100, preclamp=False)))

        decoded, _ = decode(_run(_sound(), SA3, tmp_path)[0])

        assert np.abs(decoded).max() == pytest.approx(1.0, abs=0.001)  # the 16-bit conversion clamps it, nothing else


def clip_from(samples, rate, **fields):
    from zvisiongenerator.core.audio_types import AudioClip

    return AudioClip(samples=samples, sample_rate=rate, **fields)


class TestOutOfMemory:
    @pytest.mark.parametrize(
        ("request_", "model", "message"),
        [
            (_music(), YUE2, "YuE2 3B ran out of GPU memory. Close other apps that use the GPU and try again or shorten the lyrics."),
            (_speech(), HIGGS3, "Higgs TTS 3 4B ran out of GPU memory. Close other apps that use the GPU and try again."),
            (_sound(), SA3, "Stable Audio 3 Medium ran out of GPU memory. Close other apps that use the GPU and try again."),
        ],
        ids=["music", "speech", "sound"],
    )
    @pytest.mark.parametrize("error", [torch.OutOfMemoryError("CUDA out of memory"), RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)")], ids=["cuda", "mps"])
    def test_the_message_names_the_model_and_the_backend_is_closed(self, monkeypatch, tmp_path, request_, model, message, error):
        backend = fake_backend(clip())
        backend.generate.side_effect = error
        patch_runner(monkeypatch, backend)
        monkeypatch.setattr(audio_runner, "check_speech_request", lambda *a: None)

        with pytest.raises(RuntimeError) as raised:
            _run(request_, model, tmp_path)

        assert str(raised.value) == message and raised.value.__cause__ is error
        backend.close.assert_called_once()

    def test_running_out_of_memory_while_loading_gives_the_same_message(self, monkeypatch, tmp_path):
        backend = fake_backend(clip())
        backend.load.side_effect = torch.OutOfMemoryError("out of memory")
        patch_runner(monkeypatch, backend)

        with pytest.raises(RuntimeError, match="ran out of GPU memory"):
            _run(_music(), YUE2, tmp_path)

        backend.close.assert_called_once()

    def test_another_runtime_error_is_not_rewritten(self, monkeypatch, tmp_path):
        backend = fake_backend(clip())
        backend.generate.side_effect = RuntimeError("something else broke")
        patch_runner(monkeypatch, backend)

        with pytest.raises(RuntimeError, match="^something else broke$"):
            _run(_music(), YUE2, tmp_path)


class TestClosing:
    def test_the_backend_is_closed_when_a_take_fails(self, monkeypatch, tmp_path):
        backend = fake_backend(clip())
        backend.generate.side_effect = ValueError("bad lyrics")
        patch_runner(monkeypatch, backend)

        with pytest.raises(ValueError, match="bad lyrics"):
            _run(_music(), YUE2, tmp_path)

        assert backend.events == ["load", "close"]

    def test_the_backend_is_closed_when_loading_fails(self, monkeypatch, tmp_path):
        backend = fake_backend(clip())
        backend.load.side_effect = FileNotFoundError("Higgs TTS 3 needs bosonai/higgs-tts-3-4b, which is not downloaded.")
        patch_runner(monkeypatch, backend)

        with pytest.raises(FileNotFoundError):
            _run(_speech(), HIGGS3, tmp_path)

        backend.close.assert_called_once()

    def test_the_backend_is_closed_when_the_user_quits_a_take(self, monkeypatch, tmp_path):
        control = SkipSignal()
        events: list[dict] = []
        backend = fake_backend(clip())

        def stop(request, on_progress, cancelled):
            control.queue_action("quit")
            assert cancelled()
            raise AudioCancelled

        backend.generate.side_effect = stop
        patch_runner(monkeypatch, backend)

        saved = _run(_music(), YUE2, tmp_path, runs=3, skip_signal=control, progress_callback=events.append)

        assert saved == [] and backend.generate.call_count == 1
        backend.close.assert_called_once()
        assert events[-1]["type"] == "batch_cancelled"

    def test_quit_before_the_first_take_makes_none(self, monkeypatch, tmp_path):
        control = SkipSignal()
        control.queue_action("quit")
        backend = fake_backend(clip())
        patch_runner(monkeypatch, backend)

        assert _run(_music(), YUE2, tmp_path, skip_signal=control) == []
        assert backend.generate.call_count == 0 and backend.close.call_count == 1


class TestJobControls:
    @REAL
    def test_next_drops_the_running_take_and_goes_on_with_the_next(self, monkeypatch, tmp_path):
        control = SkipSignal()
        events: list[dict] = []
        backend = fake_backend(clip(0.2))
        calls = {"n": 0}
        good = backend.generate.side_effect

        def generate(request, on_progress, cancelled):
            calls["n"] += 1
            if calls["n"] == 1:
                control.queue_action("skip")
                raise AudioCancelled
            return good(request, on_progress, cancelled)

        backend.generate.side_effect = generate
        patch_runner(monkeypatch, backend)

        saved = _run(_music(), YUE2, tmp_path, runs=2, skip_signal=control, progress_callback=events.append)

        assert len(saved) == 1
        assert [e["status"] for e in events if e["type"] == "generation_finished"] == ["skipped", "success"]
        assert events[-1]["type"] == "batch_completed"

    @REAL
    def test_repeat_makes_the_take_again_with_a_new_seed(self, monkeypatch, tmp_path):
        control = SkipSignal()
        backend = fake_backend(clip(0.2))
        first = backend.generate.side_effect
        calls = {"n": 0}

        def generate(request, on_progress, cancelled):
            calls["n"] += 1
            if calls["n"] == 1:
                control.queue_action("repeat")
            return first(request, on_progress, cancelled)

        backend.generate.side_effect = generate
        patch_runner(monkeypatch, backend)

        saved = _run(_music(), YUE2, tmp_path, seed=1, skip_signal=control)

        seeds = [call.args[0].seed for call in backend.generate.call_args_list]
        assert len(saved) == 2 and seeds[0] == 1 and seeds[1] != 1

    @REAL
    def test_pause_waits_for_resume_between_takes(self, monkeypatch, tmp_path):
        control = SkipSignal()
        backend = fake_backend(clip(0.2))
        paused = MagicMock(return_value=None)
        monkeypatch.setattr(audio_runner, "pause_until_resumed", paused)
        first = backend.generate.side_effect

        def generate(request, on_progress, cancelled):
            control.queue_action("pause")
            return first(request, on_progress, cancelled)

        backend.generate.side_effect = generate
        patch_runner(monkeypatch, backend)

        _run(_music(), YUE2, tmp_path, runs=2, skip_signal=control)

        assert paused.call_count == 2 and paused.call_args.kwargs["mode"] == "audio"

    @REAL
    def test_events_follow_the_batch_and_carry_the_backends_progress(self, monkeypatch, tmp_path):
        events: list[dict] = []
        backend = fake_backend(clip(0.2))
        original = backend.generate.side_effect

        def generate(request, on_progress, cancelled):
            on_progress("Plan", 1, None, 1, 1)
            return original(request, on_progress, cancelled)

        backend.generate.side_effect = generate
        patch_runner(monkeypatch, backend)

        saved = _run(_music(), YUE2, tmp_path, progress_callback=events.append)

        assert [e["type"] for e in events] == ["batch_started", "prompt_started", "generation_started", "audio_progress", "generation_finished", "batch_completed"]
        progress = events[3]
        assert (progress["phase"], progress["step"], progress["total"], progress["part"], progress["n_parts"]) == ("Plan", 1, None, 1, 1)
        assert events[4]["output_path"] == str(saved[0]) and all(e["mode"] == "audio" for e in events)


def _tone(seconds: float, amplitude: float, lead: float, trail: float) -> torch.Tensor:
    body = amplitude * np.sin(2 * np.pi * 220 * np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE)
    return torch.from_numpy(np.concatenate([np.zeros(round(lead * SAMPLE_RATE)), body, np.zeros(round(trail * SAMPLE_RATE))]).astype(np.float32))


def _silent_runs(wave: np.ndarray, rate: int, minimum: float = 0.05) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` samples of every run of near-silence at least *minimum* seconds long."""
    quiet = np.abs(wave) < 1e-3
    runs, start = [], None
    for index, flag in enumerate(np.append(quiet, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if (index - start) / rate >= minimum:
                runs.append((start, index))
            start = None
    return runs


def _db(value: float) -> float:
    return 20 * np.log10(value)


class TestSpeechJoinedThroughTheRunner:
    @REAL
    def test_three_parts_are_joined_with_short_edges_300_ms_pauses_and_a_level_raised_by_only_3_db(self, monkeypatch, tmp_path):
        parts = [_tone(2.0, 0.4, lead=0.4, trail=0.4), _tone(2.0, 0.4, lead=0.05, trail=1.9), _tone(2.0, 0.2, lead=0.05, trail=0.05)]
        install_fakes(monkeypatch, tmp_path, FakeModel([10, 10, 10]), FakeCodec([], list(parts)))
        patch_runner(monkeypatch, Higgs3Backend())
        text = " ".join(letter * 299 + "." for letter in "abc")  # three sentences of 300 characters: three parts

        saved = _run(_speech(text), HIGGS3, tmp_path)

        out, rate = decode(saved[0])
        wave = out[0]
        assert rate == SAMPLE_RATE and read_config(saved[0])["generation"]["parts"] == 3 and read_config(saved[0])["prompt"] == text
        first_sound = int(np.argmax(np.abs(wave) > 1e-3))
        last_sound = len(wave) - 1 - int(np.argmax(np.abs(wave[::-1]) > 1e-3))
        assert first_sound / rate == pytest.approx(0.15, abs=0.003)  # 0.4 s of lead cut to 150 ms
        assert (len(wave) - 1 - last_sound) / rate == pytest.approx(0.05, abs=0.003)
        gaps = [(end - start) / rate for start, end in _silent_runs(wave, rate) if start > 0 and end < len(wave)]
        assert gaps == [pytest.approx(0.15 + 0.3 + 0.05, abs=0.01), pytest.approx(0.15 + 0.3 + 0.05, abs=0.01)]  # trimmed tail + 300 ms pause + lead; the 1.9 s tail is gone
        cuts = [(start, end) for start, end in _silent_runs(wave, rate) if start > 0 and end < len(wave)]
        segments = [wave[first_sound : cuts[0][0]], wave[cuts[0][1] : cuts[1][0]], wave[cuts[1][1] : last_sound + 1]]
        inputs = [float(np.sqrt(np.mean(np.square(part.numpy()[np.abs(part.numpy()) > 1e-3])))) for part in parts]
        outputs = [float(np.sqrt(np.mean(np.square(segment[np.abs(segment) > 1e-3])))) for segment in segments]
        change = [_db(out_) - _db(in_) for out_, in_ in zip(outputs, inputs, strict=True)]
        assert change[2] == pytest.approx(3.0, abs=0.3)  # the quiet part is 4 dB under the mean but is raised by 3 dB at most
        assert change[0] == pytest.approx(-2.0, abs=0.3) and change[1] == pytest.approx(-2.0, abs=0.3)
