"""Tests for the voice design runner with a fake designer backend: candidates, the batch record, Quit, failure, closing and the files it writes."""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests._fake_audio import read_fake_audio
from tests._speech_fakes import RATE, tone
from tests.audio._helpers import patch_runner, preset

from zvisiongenerator.audio_runner import run_voice_design
from zvisiongenerator.core.audio_backend import AudioCancelled
from zvisiongenerator.core.audio_types import VoiceDesignRequest
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.voice_candidates import candidate_file, candidates_dir, read_batches, write_candidate

QWEN = preset("qwen3_tts")
CONFIG = {"generation": {"seed_min": 10, "seed_max": 10_000}}
REQUEST = VoiceDesignRequest(description="A calm, low voice.", text="Hello there, nice to meet you.")


class FakeDesigner:
    """A backend that can design voices: it returns *waves* in turn (the last repeats), notes its steps in *events* and can fail or queue a Quit."""

    def __init__(self, *waves: np.ndarray, events: list[str] | None = None, fail_on: int | None = None, quit_after: int | None = None, skip: SkipSignal | None = None) -> None:
        self.waves = list(waves) or [tone(1.0)]
        self.events = events if events is not None else []
        self.fail_on, self.quit_after, self.skip = fail_on, quit_after, skip
        self.sample_rate = RATE
        self.seeds: list[int] = []
        self.calls: list[tuple[str, str]] = []
        self.loaded_with: Any = None

    def load(self, preset, platform_key, cancelled=None) -> None:
        self.events.append("load")
        self.loaded_with = preset

    def close(self) -> None:
        self.events.append("close")

    def design_voice(self, description: str, text: str, seed: int) -> np.ndarray:
        self.events.append("design")
        self.seeds.append(seed)
        self.calls.append((description, text))
        count = len(self.seeds)
        if self.fail_on == count:
            raise ValueError("boom")
        if self.quit_after == count and self.skip is not None:
            self.skip.queue_action("quit")
        return self.waves.pop(0) if len(self.waves) > 1 else self.waves[0]


class NotADesigner:
    """A backend without ``design_voice``."""

    def load(self, *args: Any, **kwargs: Any) -> None: ...

    def close(self) -> None: ...

    def generate(self, *args: Any, **kwargs: Any) -> None: ...


def _design(tmp_path, monkeypatch, backend, *, request=REQUEST, present=True, skip=None, progress=None):
    env = patch_runner(monkeypatch, backend, present=present)
    batch = None
    error = None
    try:
        batch = run_voice_design(request, QWEN, CONFIG, out_dir=tmp_path / "out", progress_callback=progress, skip_signal=skip)
    except Exception as exc:
        error = exc
    return env, batch, error


def _kinds(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def test_three_candidates_are_made_with_distinct_seeds_and_the_batch_is_recorded_after_them(tmp_path, monkeypatch):
    random.seed(1234)  # three draws from 10..10000 collide in about 0.03% of runs
    backend = FakeDesigner(tone(1.0), tone(1.5), tone(2.0))

    _, batch, error = _design(tmp_path, monkeypatch, backend)

    assert error is None
    assert len(backend.seeds) == 3
    assert len(set(backend.seeds)) == 3
    assert all(10 <= seed <= 10_000 for seed in backend.seeds)
    assert backend.calls == [(REQUEST.description, REQUEST.text)] * 3
    assert batch is not None
    assert [item.seconds for item in batch.candidates] == [1.0, 1.5, 2.0]
    assert len({item.id for item in batch.candidates}) == 3
    assert (batch.model, batch.description, batch.text) == ("qwen3_tts", REQUEST.description, REQUEST.text)
    assert datetime.fromisoformat(batch.created_at).tzinfo is not None
    out = tmp_path / "out"
    assert read_batches(out) == (batch,)
    wavs = [out / f"{item.id}.wav" for item in batch.candidates]
    assert out.joinpath("candidates.yaml").stat().st_mtime_ns >= max(path.stat().st_mtime_ns for path in wavs)
    stored = read_fake_audio(candidate_file(out, batch.candidates[1].id))
    assert stored.rate == RATE
    assert stored.samples.shape == (1, round(1.5 * RATE))


def test_every_run_adds_a_batch_and_keeps_the_earlier_ones_with_their_ids(tmp_path, monkeypatch):
    _, first, _ = _design(tmp_path, monkeypatch, FakeDesigner())
    _, second, _ = _design(tmp_path, monkeypatch, FakeDesigner(), request=VoiceDesignRequest("Another voice.", "Another line.", count=2))

    assert first is not None and second is not None
    assert first.batch_id != second.batch_id
    batches = read_batches(tmp_path / "out")
    assert batches == (first, second)
    assert len({item.id for batch in batches for item in batch.candidates}) == 5
    assert sorted(path.name for path in (tmp_path / "out").glob("*.wav")) == sorted(f"{item.id}.wav" for batch in batches for item in batch.candidates)
    assert (batches[1].description, batches[1].text) == ("Another voice.", "Another line.")


def test_the_count_sets_the_number_of_candidates(tmp_path, monkeypatch):
    backend = FakeDesigner()

    _, batch, _ = _design(tmp_path, monkeypatch, backend, request=VoiceDesignRequest("d", "t", count=2))

    assert batch is not None
    assert len(batch.candidates) == 2
    assert len(backend.seeds) == 2


def test_the_design_checkpoint_is_loaded_instead_of_the_model_files(tmp_path, monkeypatch):
    backend = FakeDesigner()

    _design(tmp_path, monkeypatch, backend)

    assert backend.loaded_with.files == QWEN.design_files
    assert backend.loaded_with.files != QWEN.files


def test_the_events_come_in_order_and_progress_names_the_design_phase(tmp_path, monkeypatch):
    events: list[dict] = []

    _design(tmp_path, monkeypatch, FakeDesigner(), progress=events.append)

    assert _kinds(events) == ["model_loading", "batch_started", "audio_progress", "audio_progress", "audio_progress", "batch_completed"]
    assert events[0]["model"] == "Qwen3-TTS 1.7B VoiceDesign"
    assert events[1]["total_iterations"] == 3
    assert {event["mode"] for event in events} == {"audio"}
    progress = events[2:5]
    assert [(event["step"], event["total"]) for event in progress] == [(1, 3), (2, 3), (3, 3)]
    assert all(event["phase"] == "Design" and event["part"] == 1 and event["n_parts"] == 1 for event in progress)
    assert all(event["elapsed_secs"] >= 0 and event["eta_secs"] >= 0 for event in progress)
    assert events[-1]["completed_iterations"] == 3


def test_a_failure_after_the_first_candidate_records_it_keeps_the_earlier_batches_and_propagates(tmp_path, monkeypatch):
    _, first, _ = _design(tmp_path, monkeypatch, FakeDesigner())

    _, batch, error = _design(tmp_path, monkeypatch, FakeDesigner(fail_on=2))

    assert isinstance(error, ValueError)
    assert batch is None
    batches = read_batches(tmp_path / "out")
    assert [len(item.candidates) for item in batches] == [3, 1]
    assert batches[0] == first
    (recorded,) = batches[1].candidates
    assert (tmp_path / "out" / f"{recorded.id}.wav").is_file()
    assert len(list((tmp_path / "out").glob("*.wav"))) == 4


def test_a_load_failure_propagates_and_adds_nothing(tmp_path, monkeypatch):
    _, first, _ = _design(tmp_path, monkeypatch, FakeDesigner())
    backend = FakeDesigner()
    backend.load = MagicMock(side_effect=RuntimeError("no load"))

    _, _, error = _design(tmp_path, monkeypatch, backend)

    assert isinstance(error, RuntimeError)
    assert read_batches(tmp_path / "out") == (first,)


def test_a_quit_after_the_first_candidate_keeps_it_and_the_earlier_batches_and_ends_the_batch_cancelled(tmp_path, monkeypatch):
    _, first, _ = _design(tmp_path, monkeypatch, FakeDesigner())
    skip = SkipSignal()
    backend = FakeDesigner(quit_after=1, skip=skip)
    events: list[dict] = []

    _, batch, error = _design(tmp_path, monkeypatch, backend, skip=skip, progress=events.append)

    assert error is None
    assert batch is not None
    assert len(batch.candidates) == 1
    assert read_batches(tmp_path / "out") == (first, batch)
    assert len(backend.seeds) == 1
    assert _kinds(events) == ["model_loading", "batch_started", "audio_progress", "batch_cancelled"]
    assert events[-1]["completed_iterations"] == 1


def test_a_quit_before_the_load_makes_nothing_and_never_loads(tmp_path, monkeypatch):
    skip = SkipSignal()
    skip.queue_action("quit")
    backend = FakeDesigner()
    events: list[dict] = []

    env, batch, error = _design(tmp_path, monkeypatch, backend, skip=skip, progress=events.append)

    assert error is None
    assert batch is not None and batch.candidates == ()
    assert read_batches(tmp_path / "out") == ()
    assert "load" not in backend.events
    assert _kinds(events) == ["batch_cancelled"]
    assert backend.events == ["close"]
    env.release.assert_called_once()


def test_a_quit_while_the_model_loads_ends_the_run_cancelled(tmp_path, monkeypatch):
    backend = FakeDesigner()
    backend.load = MagicMock(side_effect=AudioCancelled())
    events: list[dict] = []

    _, batch, error = _design(tmp_path, monkeypatch, backend, progress=events.append)

    assert error is None
    assert batch is not None and batch.candidates == ()
    assert _kinds(events) == ["model_loading", "batch_cancelled"]


@pytest.mark.parametrize("case", ["success", "failure", "quit"])
def test_the_backend_is_closed_and_memory_released_last_on_every_ending(tmp_path, monkeypatch, case):
    skip = SkipSignal()
    events: list[str] = []
    backend = FakeDesigner(events=events, fail_on=2 if case == "failure" else None, quit_after=1 if case == "quit" else None, skip=skip)

    env, _, _ = _design(tmp_path, monkeypatch, backend, skip=skip)

    assert events[-1] == "close"
    assert events.count("close") == 1
    env.release.assert_called_once()
    env.session.release_resident_enhancer.assert_called_once()


def test_a_missing_download_raises_file_not_found_and_touches_nothing(tmp_path, monkeypatch):
    kept = write_candidate(tmp_path / "out", tone(1.0), RATE)
    backend = FakeDesigner()

    _, run, error = _design(tmp_path, monkeypatch, backend, present=False)

    assert isinstance(error, FileNotFoundError)
    assert backend.events == []
    assert (tmp_path / "out" / f"{kept.id}.wav").is_file()


def test_a_backend_that_cannot_design_raises_runtime_error(tmp_path, monkeypatch):
    _, _, error = _design(tmp_path, monkeypatch, NotADesigner())

    assert isinstance(error, RuntimeError)


def test_a_peak_above_full_scale_is_levelled_and_a_lower_peak_is_kept(tmp_path, monkeypatch):
    loud = tone(1.0, level=1.5)
    backend = FakeDesigner(loud, tone(1.0, level=0.3))

    _design(tmp_path, monkeypatch, backend)

    loud_candidate, quiet_candidate = read_batches(tmp_path / "out")[0].candidates[:2]
    levelled = read_fake_audio(candidate_file(tmp_path / "out", loud_candidate.id)).samples
    quiet = read_fake_audio(candidate_file(tmp_path / "out", quiet_candidate.id)).samples
    assert 0.85 < float(np.abs(levelled).max()) <= 1.0
    assert float(np.abs(quiet).max()) == pytest.approx(0.3, abs=1e-3)


def test_a_wave_without_sound_adds_no_candidate(tmp_path, monkeypatch):
    backend = FakeDesigner(tone(1.0), np.zeros(0, dtype=np.float32), tone(2.0))

    _, batch, _ = _design(tmp_path, monkeypatch, backend)

    assert batch is not None
    assert [item.seconds for item in batch.candidates] == [1.0, 2.0]
    assert len(list((tmp_path / "out").glob("*.wav"))) == 2


def test_a_run_where_every_wave_is_silent_fails_and_leaves_the_earlier_batches(tmp_path, monkeypatch):
    _, first, _ = _design(tmp_path, monkeypatch, FakeDesigner())
    events: list[dict] = []

    _, batch, error = _design(tmp_path, monkeypatch, FakeDesigner(np.zeros(0, dtype=np.float32)), progress=events.append)

    assert isinstance(error, RuntimeError)
    assert batch is None
    assert read_batches(tmp_path / "out") == (first,)
    assert "batch_completed" not in _kinds(events)


def test_nothing_is_written_outside_the_out_dir(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    _design(tmp_path, monkeypatch, FakeDesigner())

    assert sorted(path.name for path in tmp_path.iterdir()) == ["out", "work"]
    assert list(work.iterdir()) == []
    assert not candidates_dir().exists()


@pytest.mark.parametrize("stage", ["load", "design"])
def test_an_out_of_memory_error_becomes_the_friendly_message_and_adds_no_batch(tmp_path, monkeypatch, stage):
    backend = FakeDesigner()
    if stage == "load":
        backend.load = MagicMock(side_effect=torch.OutOfMemoryError("CUDA out of memory"))
    else:
        backend.design_voice = MagicMock(side_effect=torch.OutOfMemoryError("CUDA out of memory"))

    env, _, error = _design(tmp_path, monkeypatch, backend)

    assert isinstance(error, RuntimeError)
    assert "ran out of GPU memory" in str(error)
    assert ("VoiceDesign" in str(error)) == (stage == "load")
    assert read_batches(tmp_path / "out") == ()
    assert backend.events[-1] == "close"
    env.release.assert_called_once()
