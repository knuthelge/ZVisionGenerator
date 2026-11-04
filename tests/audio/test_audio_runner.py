"""Tests for the audio runner with the backend mocked: order of steps, saving, level, out of memory, closing and the job controls."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests._audio_presets import audio_config
from tests._media import REAL
from tests._torch_flags import current_flags, set_app_flags, set_kit_flags
from tests.audio._helpers import clip, decode, fake_backend, patch_runner, preset, sine

from zvisiongenerator import audio_runner
from zvisiongenerator.audio_runner import run_audio_batch
from zvisiongenerator.core.audio_backend import AudioCancelled
from zvisiongenerator.core.audio_types import AudioGenerationRequest, VoiceRef
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget
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
    def test_the_enhancer_is_released_then_the_model_loads_and_is_closed_last(self, monkeypatch, tmp_path):
        order: list[str] = []
        backend = fake_backend(clip(), events=order)
        env = patch_runner(monkeypatch, backend)
        env.session.release_resident_enhancer.side_effect = lambda: order.append("release")

        _run(_music(), YUE2, tmp_path)

        assert order == ["release", "load", "generate", "close"]

    def test_the_batch_starts_once_the_model_is_loaded(self, monkeypatch, tmp_path):
        order: list[str] = []
        backend = fake_backend(clip(), events=order)
        patch_runner(monkeypatch, backend)

        _run(_music(), YUE2, tmp_path, progress_callback=lambda event: order.append(event["type"]))

        assert order[:4] == ["model_loading", "load", "batch_started", "prompt_started"]

    def test_the_load_event_names_the_model_by_its_label(self, monkeypatch, tmp_path):
        events: list[dict] = []
        patch_runner(monkeypatch, fake_backend(clip()))

        _run(_music(), YUE2, tmp_path, progress_callback=events.append)

        loading = next(e for e in events if e["type"] == "model_loading")
        assert (loading["mode"], loading["model"]) == ("audio", "YuE2 3B")

    def test_the_backend_of_the_presets_family_loads_once_for_all_takes(self, monkeypatch, tmp_path):
        backend = fake_backend(clip())
        patch_runner(monkeypatch, backend)

        _run(_music(), YUE2, tmp_path, runs=3)

        audio_runner.get_audio_backend.assert_called_once_with("yue2")
        backend.load.assert_called_once()
        assert backend.load.call_args.args == (YUE2, "cuda")
        assert backend.generate.call_count == 3 and backend.close.call_count == 1


class TestFilesOnDisk:
    def test_files_that_are_not_on_disk_end_the_run_before_the_model_loads(self, monkeypatch, tmp_path):
        backend = fake_backend(clip())
        patch_runner(monkeypatch, backend, present=False)

        with pytest.raises(FileNotFoundError, match="is not downloaded") as raised:
            _run(_music(), YUE2, tmp_path)

        assert "YuE2 3B" in str(raised.value)
        assert backend.load.call_count == 0

    def test_files_on_disk_load_as_usual(self, monkeypatch, tmp_path):
        patch_runner(monkeypatch, fake_backend(clip()))

        assert len(_run(_music(), YUE2, tmp_path)) == 1


class TestLifecycle:
    def test_the_torch_flags_the_model_code_changed_are_given_back_once_the_backend_is_closed(self, monkeypatch, tmp_path, torch_flags_guard):
        backend = fake_backend(clip())
        backend.load.side_effect = lambda *args, **kwargs: set_kit_flags()
        patch_runner(monkeypatch, backend)
        set_app_flags()
        before = current_flags()

        _run(_music(), YUE2, tmp_path)

        assert current_flags() == before

    def test_the_torch_flags_and_the_accelerator_memory_are_given_back_when_a_take_fails(self, monkeypatch, tmp_path, torch_flags_guard):
        backend = fake_backend(clip())
        backend.load.side_effect = lambda *args, **kwargs: set_kit_flags()
        backend.generate.side_effect = ValueError("bad lyrics")
        env = patch_runner(monkeypatch, backend)
        set_app_flags()
        before = current_flags()

        with pytest.raises(ValueError, match="bad lyrics"):
            _run(_music(), YUE2, tmp_path)

        assert current_flags() == before
        env.release.assert_called_once()

    def test_the_torch_flags_are_given_back_when_loading_fails(self, monkeypatch, tmp_path, torch_flags_guard):
        def failing_load(*args, **kwargs):
            set_kit_flags()
            raise FileNotFoundError("not downloaded")

        backend = fake_backend(clip())
        backend.load.side_effect = failing_load
        env = patch_runner(monkeypatch, backend)
        set_app_flags()
        before = current_flags()

        with pytest.raises(FileNotFoundError):
            _run(_music(), YUE2, tmp_path)

        assert current_flags() == before
        backend.close.assert_called_once()
        env.release.assert_called_once()

    def test_the_memory_is_released_after_the_backend_is_closed(self, monkeypatch, tmp_path):
        order: list[str] = []
        backend = fake_backend(clip(), events=order)
        env = patch_runner(monkeypatch, backend)
        env.release.side_effect = lambda: order.append("release_memory")

        _run(_music(), YUE2, tmp_path)

        assert order[-2:] == ["close", "release_memory"]


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
    def test_the_console_says_when_a_take_was_cut_at_the_models_limit_and_only_then(self, monkeypatch, tmp_path, capsys):
        patch_runner(monkeypatch, fake_backend(clip(0.2, cut=True), clip(0.2)))

        _run(_music(), YUE2, tmp_path, runs=2)

        assert capsys.readouterr().out.count(audio_runner._CUT_NOTE) == 1

    @REAL
    def test_a_speech_take_records_the_full_text_and_the_number_of_parts(self, monkeypatch, tmp_path):
        patch_runner(monkeypatch, fake_backend(clip(0.2)))
        sentence = "This sentence has a fair number of characters in it, give or take. "
        text = sentence * 20  # about 1,340 characters: three parts of up to 500

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
    def test_a_take_that_peaks_over_full_scale_is_brought_to_minus_1_dbfs(self, monkeypatch, tmp_path):
        loud = sine(0.3, rate=44100, amplitude=10 ** (6.3 / 20))
        patch_runner(monkeypatch, fake_backend(clip_from(loud, 44100)))

        saved = _run(_sound(), SA3, tmp_path)

        decoded, _ = decode(saved[0])
        assert 20 * np.log10(np.abs(decoded).max()) == pytest.approx(-1.0, abs=0.01)

    @REAL
    def test_a_take_that_stays_within_full_scale_is_saved_as_it_is(self, monkeypatch, tmp_path):
        samples = sine(0.3, rate=44100, amplitude=0.95)
        patch_runner(monkeypatch, fake_backend(clip_from(samples, 44100)))

        decoded, _ = decode(_run(_sound(), SA3, tmp_path)[0])

        assert np.abs(decoded).max() == pytest.approx(0.95, abs=0.001)


def clip_from(samples, rate, **fields):
    from zvisiongenerator.core.audio_types import AudioClip

    return AudioClip(samples=samples, sample_rate=rate, **fields)


class TestTiming:
    @staticmethod
    def _eta(*, gen_times, takes_left, ran_secs):
        events: list[dict] = []
        console = MagicMock()
        report = audio_runner._progress_reporter(
            events.append, {"mode": "audio"}, console, batch_start=time.time() - 100, take_began=time.monotonic() - ran_secs, gen_times=gen_times, takes_left=takes_left
        )

        report("Sampling", 1, 2, 1, 1)

        (event,) = events
        return event

    def test_the_time_left_counts_the_takes_to_go_less_what_this_one_has_run(self):
        event = self._eta(gen_times=[20.0, 40.0], takes_left=3, ran_secs=10)

        assert event["eta_secs"] == pytest.approx(80.0, abs=1.0)  # 30 s on average x 3 takes - 10 s
        assert event["elapsed_secs"] == pytest.approx(100.0, abs=1.0)

    def test_the_time_left_is_never_below_zero(self):
        assert self._eta(gen_times=[20.0], takes_left=1, ran_secs=500)["eta_secs"] == 0.0

    def test_there_is_no_estimate_before_a_take_has_finished(self):
        assert self._eta(gen_times=[], takes_left=2, ran_secs=5)["eta_secs"] is None

    @REAL
    def test_the_second_take_of_a_job_counts_down_from_the_first_ones_time(self, monkeypatch, tmp_path):
        events: list[dict] = []
        backend = fake_backend(clip(0.1))
        original = backend.generate.side_effect

        def generate(request, on_progress, cancelled):
            on_progress("Sampling", 1, 1, 1, 1)
            return original(request, on_progress, cancelled)

        backend.generate.side_effect = generate
        patch_runner(monkeypatch, backend)

        _run(_music(), YUE2, tmp_path, runs=2, progress_callback=events.append)

        first, second = [e for e in events if e["type"] == "audio_progress"]
        assert first["eta_secs"] is None and second["eta_secs"] is not None and second["eta_secs"] >= 0.0
        assert all("elapsed_secs" in e for e in (first, second))


class TestConsole:
    @REAL
    def test_the_first_take_header_follows_the_loading_line_and_later_takes_are_set_off_by_a_blank_row(self, monkeypatch, tmp_path, capsys):
        patch_runner(monkeypatch, fake_backend(clip(0.2), clip(0.2)))

        _run(_music(), YUE2, tmp_path, runs=2)

        lines = capsys.readouterr().out.splitlines()
        first, second = (index for index, line in enumerate(lines) if line.startswith("["))
        assert lines[first - 1].startswith("Loading ")
        assert lines[second - 1] == ""

    @REAL
    def test_a_phase_is_named_once_on_a_console_that_is_not_a_terminal(self, monkeypatch, tmp_path, capsys):
        backend = fake_backend(clip(0.2))
        original = backend.generate.side_effect

        def generate(request, on_progress, cancelled):
            for step in range(1, 4):
                on_progress("Song tokens", step, None, 1, 1)
            return original(request, on_progress, cancelled)

        backend.generate.side_effect = generate
        patch_runner(monkeypatch, backend)

        _run(_music(), YUE2, tmp_path)

        assert capsys.readouterr().out.count("Song tokens") == 1


class TestSeeds:
    def test_a_fixed_seed_is_every_takes_seed_and_a_missing_one_is_drawn_per_take_within_the_configured_range(self, monkeypatch, tmp_path):
        backend = fake_backend(clip(0.2))
        patch_runner(monkeypatch, backend)
        config = {**CONFIG, "generation": {"seed_min": 100, "seed_max": 102}}

        run_audio_batch(_music(), YUE2, config, runs=20, seed=None, output_dir=str(tmp_path / "out"))

        drawn = {call.args[0].seed for call in backend.generate.call_args_list}
        assert drawn <= {100, 101, 102} and len(drawn) > 1

    def test_the_takes_are_numbered_for_the_job_events(self, monkeypatch, tmp_path):
        events: list[dict] = []
        patch_runner(monkeypatch, fake_backend(clip(0.2)))

        _run(_music(), YUE2, tmp_path, runs=2, progress_callback=events.append)

        started = [e for e in events if e["type"] == "prompt_started"]
        assert [(e["run_index"], e["ran_iterations"], e["total_iterations"], e["total_runs"]) for e in started] == [(0, 1, 2, 2), (1, 2, 2, 2)]


class TestOutOfMemory:
    @pytest.mark.parametrize(
        ("request_", "model", "model_name"),
        [
            (_music(), YUE2, "YuE2 3B"),
            (_speech(), HIGGS3, "Higgs TTS 3 4B"),
            (_sound(), SA3, "Stable Audio 3 Medium"),
        ],
        ids=["music", "speech", "sound"],
    )
    @pytest.mark.parametrize("error", [torch.OutOfMemoryError("CUDA out of memory"), RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)")], ids=["cuda", "mps"])
    def test_the_message_names_the_model_and_the_backend_is_closed(self, monkeypatch, tmp_path, request_, model, model_name, error):
        backend = fake_backend(clip())
        backend.generate.side_effect = error
        patch_runner(monkeypatch, backend)

        with pytest.raises(RuntimeError) as raised:
            _run(request_, model, tmp_path)

        assert model_name in str(raised.value) and "GPU memory" in str(raised.value)
        assert raised.value.__cause__ is error
        backend.close.assert_called_once()

    def test_running_out_of_memory_while_loading_gives_the_same_message(self, monkeypatch, tmp_path):
        backend = fake_backend(clip())
        backend.load.side_effect = torch.OutOfMemoryError("out of memory")
        patch_runner(monkeypatch, backend)

        with pytest.raises(RuntimeError, match="ran out of GPU memory"):
            _run(_music(), YUE2, tmp_path)

        backend.close.assert_called_once()

    def test_the_message_names_what_the_model_needs_against_the_cards_size_when_known(self, monkeypatch, tmp_path):
        backend = fake_backend(clip())
        backend.generate.side_effect = torch.OutOfMemoryError("out of memory")
        patch_runner(monkeypatch, backend)
        monkeypatch.setattr(audio_runner, "accelerator_memory_budget_or_none", lambda: MemoryBudget(8 * GIB, system_bytes=64 * GIB))

        with pytest.raises(RuntimeError) as raised:
            _run(_speech(), HIGGS3, tmp_path)

        assert "8.2 GB" in str(raised.value) and "8 GB" in str(raised.value)

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


class TestQuitWhileLoading:
    def test_quit_queued_before_the_load_never_loads_the_model(self, monkeypatch, tmp_path):
        control = SkipSignal()
        control.queue_action("quit")
        events: list[dict] = []
        backend = fake_backend(clip())
        patch_runner(monkeypatch, backend)

        saved = _run(_music(), YUE2, tmp_path, runs=2, skip_signal=control, progress_callback=events.append)

        assert saved == [] and backend.load.call_count == 0 and backend.generate.call_count == 0
        assert [e["type"] for e in events] == ["batch_cancelled"]
        assert events[0] == {"type": "batch_cancelled", "mode": "audio", "completed_iterations": 0, "total_iterations": 2}

    def test_the_load_gets_a_check_that_follows_quit(self, monkeypatch, tmp_path):
        control = SkipSignal()
        backend = fake_backend(clip())
        seen: list[bool] = []

        def load(model, device, cancelled=None):
            seen.append(cancelled())
            control.queue_action("quit")
            seen.append(cancelled())

        backend.load.side_effect = load
        patch_runner(monkeypatch, backend)

        _run(_music(), YUE2, tmp_path, skip_signal=control)

        assert seen == [False, True]

    def test_skip_does_not_count_as_quit_for_the_load(self, monkeypatch, tmp_path):
        control = SkipSignal()
        backend = fake_backend(clip())
        seen: list[bool] = []

        def load(model, device, cancelled=None):
            control.queue_action("skip")
            seen.append(cancelled())

        backend.load.side_effect = load
        patch_runner(monkeypatch, backend)

        _run(_music(), YUE2, tmp_path, skip_signal=control)

        assert seen == [False]

    def test_a_stop_during_the_load_closes_the_backend_and_reports_a_cancelled_batch(self, monkeypatch, tmp_path):
        control = SkipSignal()
        events: list[dict] = []
        backend = fake_backend(clip())

        def load(model, device, cancelled=None):
            control.queue_action("quit")
            assert cancelled()
            raise AudioCancelled

        backend.load.side_effect = load
        env = patch_runner(monkeypatch, backend)

        saved = _run(_music(), YUE2, tmp_path, runs=3, skip_signal=control, progress_callback=events.append)

        assert saved == [] and backend.generate.call_count == 0
        backend.close.assert_called_once()
        env.release.assert_called_once()
        assert [e["type"] for e in events] == ["model_loading", "batch_cancelled"]
        assert events[-1] == {"type": "batch_cancelled", "mode": "audio", "completed_iterations": 0, "total_iterations": 3}


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
    @pytest.mark.parametrize(("queued_on_resume", "takes"), [("repeat", 2), ("skip", 1), (None, 1)])
    def test_a_pause_after_a_take_then_honours_what_was_queued_while_paused(self, monkeypatch, tmp_path, queued_on_resume, takes):
        control = SkipSignal()
        events: list[dict] = []
        backend = fake_backend(clip(0.2))
        first = backend.generate.side_effect
        calls = {"n": 0}
        resumed = iter([queued_on_resume, None])
        monkeypatch.setattr(audio_runner, "pause_until_resumed", MagicMock(side_effect=lambda *_a, **_k: next(resumed)))

        def generate(request, on_progress, cancelled):
            calls["n"] += 1
            if calls["n"] == 1:
                control.queue_action("pause")
            return first(request, on_progress, cancelled)

        backend.generate.side_effect = generate
        patch_runner(monkeypatch, backend)

        saved = _run(_music(), YUE2, tmp_path, seed=1, skip_signal=control, progress_callback=events.append)

        assert len(saved) == takes
        assert [e["type"] for e in events].count("generation_repeat") == takes - 1

    @REAL
    def test_a_quit_queued_while_paused_after_a_take_stops_the_batch(self, monkeypatch, tmp_path):
        control = SkipSignal()
        events: list[dict] = []
        backend = fake_backend(clip(0.2))
        first = backend.generate.side_effect

        def generate(request, on_progress, cancelled):
            control.queue_action("pause")
            return first(request, on_progress, cancelled)

        backend.generate.side_effect = generate
        monkeypatch.setattr(audio_runner, "pause_until_resumed", MagicMock(return_value="quit"))
        patch_runner(monkeypatch, backend)

        saved = _run(_music(), YUE2, tmp_path, runs=2, skip_signal=control, progress_callback=events.append)

        assert len(saved) == 1 and events[-1]["type"] == "batch_cancelled"

    @REAL
    def test_the_batch_reports_the_takes_made_against_the_takes_planned_including_repeats(self, monkeypatch, tmp_path):
        control = SkipSignal()
        events: list[dict] = []
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

        _run(_music(), YUE2, tmp_path, runs=2, skip_signal=control, progress_callback=events.append)

        done = events[-1]
        assert (done["type"], done["completed_iterations"], done["total_iterations"]) == ("batch_completed", 3, 3)

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

        assert [e["type"] for e in events] == ["model_loading", "batch_started", "prompt_started", "generation_started", "audio_progress", "generation_finished", "batch_completed"]
        progress = events[4]
        assert (progress["phase"], progress["step"], progress["total"], progress["part"], progress["n_parts"]) == ("Plan", 1, None, 1, 1)
        assert events[5]["output_path"] == str(saved[0]) and all(e["mode"] == "audio" for e in events)
