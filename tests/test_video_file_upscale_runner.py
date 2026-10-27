"""Tests for upscaling an existing video file: the source check, the event flow, sound, cancelling and cleanup.

ffmpeg, ffprobe and the engine are replaced; nothing here decodes video or loads a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests._factories import make_extend_source, make_probe

from zvisiongenerator import video_file_upscale_runner as runner
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import VideoFileUpscaleRequest
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.upscale import VIDEO_UPSCALE_SEED, WORK_DIR_PREFIX
from zvisiongenerator.utils.video_extend import ExtendInspection, ExtendProblem, extend_source_problems
from zvisiongenerator.video_file_upscale_runner import read_video_upscale_source, run_video_file_upscale

_FINISHED = "finished.mp4"
_RECORDED_CONFIG = {"schema": "zvisiongenerator.config.v1", "workflow": "txt2vid", "prompt": "A knight rides in.", "model": "ltx-8", "seed": 7, "steps": 8, "frame_count": 7}


def _frames(count: int) -> list[np.ndarray]:
    return [np.full((2, 4, 3), i, dtype=np.uint8) for i in range(count)]


class _FakeEngine:
    """A SeedVR2 engine whose ``upscale_video`` yields one frame per input frame and reports one batch per ``batch_size`` frames."""

    def __init__(self, clock: _Clock | None = None, seconds_per_batch: float = 0.0) -> None:
        self.calls: list[dict[str, Any]] = []
        self.finished = 0
        self.closed = False
        self._clock = clock
        self._seconds = seconds_per_batch
        self.after_frames = None
        self.raise_at: int | None = None

    def upscale_video(self, model, frames, *, width, height, total_frames, seed, softness, batch_size, progress, interrupt):
        self.calls.append({"model": model, "width": width, "height": height, "total_frames": total_frames, "seed": seed, "softness": softness, "batch_size": batch_size})
        return self._stream(frames, batch_size, -(-total_frames // batch_size), progress, interrupt)

    def _stream(self, frames, batch_size, total_batches, progress, interrupt):
        done_frames = 0
        batches = 0
        try:
            for frame in frames:
                if done_frames == self.raise_at:
                    raise RuntimeError("engine blew up")
                done_frames += 1
                yield frame
                if done_frames % batch_size == 0:
                    batches += 1
                    if self._clock is not None:
                        self._clock.now += self._seconds
                    progress(batches, total_batches)
                    interrupt()
            if self.after_frames is not None:
                self.after_frames()
        finally:
            self.finished += 1
            self.closed = True


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    """Replace ffmpeg and the frame reader; record every ffmpeg command and the frames written."""
    state = SimpleNamespace()
    state.ffmpeg = []
    state.on_finish = None
    state.written = []
    state.reader_closed = False
    state.source_frames = _frames(7)
    state.writer_stops_after = None
    state.writer_error = None
    state.ffmpeg_fail_on = None

    def run_ffmpeg(command, message="ffmpeg failed"):
        state.ffmpeg.append(command)
        if state.ffmpeg_fail_on and state.ffmpeg_fail_on in command[-1]:
            raise RuntimeError(f"{message}: boom")
        Path(command[-1]).write_bytes(b"media")
        if state.on_finish and command[-1].endswith(_FINISHED):
            state.on_finish()

    def stream_frames(path, *, width, height):
        try:
            yield from state.source_frames
        finally:
            state.reader_closed = True

    def write_segment_frames(frames, command, *, description):
        for frame in frames:
            state.written.append(frame)
            if state.writer_stops_after is not None and len(state.written) >= state.writer_stops_after:
                break
        if state.writer_error:
            raise RuntimeError(state.writer_error)
        state.segment_command = command
        Path(command[-1]).write_bytes(b"video")

    monkeypatch.setattr(runner, "run_ffmpeg", run_ffmpeg)
    monkeypatch.setattr(runner, "stream_frames", stream_frames)
    monkeypatch.setattr(runner, "write_segment_frames", write_segment_frames)
    return state


def _request(tmp_path, *, has_audio=True, audio_codec="aac", frame_count=7, batch_size=1, recorded_config=_RECORDED_CONFIG, **source_overrides) -> VideoFileUpscaleRequest:
    source = make_extend_source(
        path=tmp_path / "clip.mp4",
        width=4,
        height=2,
        frame_count=frame_count,
        has_audio=has_audio,
        video_start=0.021333,
        probe=make_probe(has_audio=has_audio, audio_codec=audio_codec if has_audio else None),
        **source_overrides,
    )
    return VideoFileUpscaleRequest(source=source, recorded_config=recorded_config, output_dir=str(tmp_path), width=8, height=4, upscale_model="seedvr2-7b", softness=0.0, batch_size=batch_size)


def _run(tmp_path, fakes, engine=None, *, skip=None, request=None, clock=None):
    events: list[dict[str, Any]] = []
    engine = engine or _FakeEngine()
    outcome = run_video_file_upscale(engine, "model", request or _request(tmp_path), progress_callback=events.append, skip_signal=skip, **({"clock": clock} if clock else {}))
    return outcome, events, engine


def _types(events):
    return [event["type"] for event in events]


def _finish_command(fakes) -> list[str]:
    """Return the ffmpeg command that wrote the finished file: the last step, which carries the audio and the settings."""
    return next(command for command in fakes.ffmpeg if command[-1].endswith(_FINISHED))


def _embedded(fakes) -> dict[str, Any]:
    """Return the config payload that the finishing command stores as MP4 metadata."""
    command = _finish_command(fakes)
    key = "zvisiongenerator.config="
    return json.loads(next(part for part in command if part.startswith(key))[len(key) :])


def _leftovers(tmp_path) -> list[str]:
    return sorted(path.name for path in tmp_path.iterdir() if path.name != "clip.mp4")


class TestSuccess:
    def test_the_events_run_in_order_and_the_output_sits_beside_the_source(self, tmp_path, fakes):
        outcome, events, _engine = _run(tmp_path, fakes)

        assert outcome is StageOutcome.success
        steps = [event for event in events if event["type"] == "step_progress"]
        assert [t for t in _types(events) if t != "step_progress"] == [
            "batch_started",
            "prompt_started",
            "generation_started",
            "workflow_started",
            "workflow_stage_started",
            "workflow_stage_completed",
            "workflow_finished",
            "generation_finished",
            "batch_completed",
        ]
        assert _types(events).index("step_progress") > _types(events).index("workflow_stage_started")
        assert len(steps) == 8
        assert all(event["mode"] == "video" for event in events)
        started = next(event for event in events if event["type"] == "workflow_started")
        assert (started["stage_names"], started["total_stages"]) == (["upscale"], 1)
        finished = next(event for event in events if event["type"] == "generation_finished")
        assert finished["status"] == "success"
        assert Path(finished["output_path"]) == tmp_path / "clip_2x.mp4"
        assert finished["filename"] == "clip_2x.mp4"
        assert _leftovers(tmp_path) == ["clip_2x.mp4"]

    def test_the_engine_gets_the_frames_the_output_size_and_the_batch_plan_and_is_closed_after(self, tmp_path, fakes):
        request = _request(tmp_path, batch_size=3)

        _outcome, _events, engine = _run(tmp_path, fakes, request=request)

        assert engine.calls == [{"model": "model", "width": 8, "height": 4, "total_frames": 7, "seed": VIDEO_UPSCALE_SEED, "softness": 0.0, "batch_size": 3}]
        assert len(fakes.written) == 7
        assert engine.finished == 1
        assert fakes.reader_closed is True

    def test_the_encoder_gets_the_output_size_the_source_frame_rate_and_colour_tags(self, tmp_path, fakes):
        request = _request(tmp_path, fps_text="30000/1001", color_space="bt709", color_primaries="bt709", color_trc="bt709")

        _run(tmp_path, fakes, request=request)

        command = fakes.segment_command
        assert command[command.index("-s") + 1] == "8x4"
        assert command[command.index("-r") + 1] == "30000/1001"
        assert "colorspace=bt709" in command[command.index("-vf") + 1]

    def test_the_result_embeds_the_sources_settings_and_the_upscale(self, tmp_path, fakes):
        _run(tmp_path, fakes)

        payload = _embedded(fakes)
        assert (payload["workflow"], payload["prompt"], payload["seed"]) == ("upscale", "A knight rides in.", 7)
        assert (payload["width"], payload["height"], payload["frame_count"]) == (8, 4, 7)
        assert payload["source"] == {"path": str(tmp_path / "clip.mp4"), "width": 4, "height": 2, "workflow": "txt2vid"}
        assert payload["generation"]["audio"] is True
        assert payload["generation"]["upscale"] == {"method": "seedvr2", "factor": 2, "model": "seedvr2-7b", "quantize": 8, "softness": 0.0, "seed": VIDEO_UPSCALE_SEED}
        assert Path(_finish_command(fakes)[-1]).parent.name.startswith(WORK_DIR_PREFIX)

    def test_the_recorded_frame_count_is_the_frames_written(self, tmp_path, fakes):
        fakes.source_frames = _frames(9)

        _run(tmp_path, fakes)

        assert _embedded(fakes)["frame_count"] == 9

    def test_a_taken_name_gets_a_counter(self, tmp_path, fakes):
        (tmp_path / "clip_2x.mp4").write_bytes(b"older")

        _outcome, events, _engine = _run(tmp_path, fakes)

        finished = next(event for event in events if event["type"] == "generation_finished")
        assert Path(finished["output_path"]).name == "clip_2x_2.mp4"
        assert (tmp_path / "clip_2x.mp4").read_bytes() == b"older"


class TestSound:
    def _prep(self, fakes):
        return next((c for c in fakes.ffmpeg if c[-1].endswith("audio.m4a")), None)

    def test_an_aac_track_is_copied_and_muxed_with_the_sources_video_offset(self, tmp_path, fakes):
        _run(tmp_path, fakes)

        mux = _finish_command(fakes)
        prep = self._prep(fakes)
        assert prep[prep.index("-c:a") + 1] == "copy"
        assert mux[mux.index("-itsoffset") + 1] == "0.021333"
        assert mux[mux.index("-c") + 1] == "copy"
        assert "-shortest" not in mux

    def test_any_other_track_is_encoded_as_aac(self, tmp_path, fakes):
        _run(tmp_path, fakes, request=_request(tmp_path, audio_codec="opus"))

        prep = self._prep(fakes)
        assert (prep[prep.index("-c:a") + 1], prep[prep.index("-b:a") + 1]) == ("aac", "192k")

    def test_a_clip_without_sound_runs_no_audio_step_and_records_it(self, tmp_path, fakes):
        outcome, _events, _engine = _run(tmp_path, fakes, request=_request(tmp_path, has_audio=False))

        assert outcome is StageOutcome.success
        assert self._prep(fakes) is None
        assert _finish_command(fakes).count("-i") == 1
        assert _embedded(fakes)["generation"]["audio"] is False

    def test_a_broken_track_fails_the_job_before_the_engine_starts(self, tmp_path, fakes):
        fakes.ffmpeg_fail_on = "audio.m4a"

        outcome, events, engine = _run(tmp_path, fakes)

        assert outcome is StageOutcome.failed
        assert engine.calls == []
        assert not [event for event in events if event["type"] == "step_progress"]
        assert _types(events)[-1] == "batch_failed"
        assert _leftovers(tmp_path) == []


class TestCancelling:
    def test_a_quit_before_the_start_cancels_without_touching_anything(self, tmp_path, fakes):
        skip = SkipSignal()
        skip.queue_action("quit")

        outcome, events, engine = _run(tmp_path, fakes, skip=skip)

        assert outcome is StageOutcome.skipped
        assert _types(events) == ["batch_started", "prompt_started", "batch_cancelled"]
        assert engine.calls == [] and fakes.ffmpeg == []
        assert _leftovers(tmp_path) == []

    def test_a_quit_in_the_middle_of_the_stream_stops_the_engine_and_leaves_nothing(self, tmp_path, fakes):
        skip = SkipSignal()
        engine = _FakeEngine()
        original = engine._stream

        def quit_after_three(frames, batch_size, total_batches, progress, interrupt):
            for index, frame in enumerate(original(frames, batch_size, total_batches, progress, interrupt)):
                if index == 2:
                    skip.queue_action("quit")
                yield frame

        engine._stream = quit_after_three

        outcome, events, engine = _run(tmp_path, fakes, engine, skip=skip)

        assert outcome is StageOutcome.skipped
        assert len(fakes.written) < 7
        assert _types(events)[-2:] == ["generation_finished", "batch_cancelled"]
        assert next(event for event in events if event["type"] == "generation_finished")["status"] == "skipped"
        assert engine.finished == 1 and fakes.reader_closed
        assert _leftovers(tmp_path) == []

    @pytest.mark.parametrize("moment", ["after_the_last_batch", "while_finishing"])
    def test_a_late_quit_still_cancels_and_leaves_nothing(self, tmp_path, fakes, moment):
        skip = SkipSignal()
        engine = _FakeEngine()
        if moment == "after_the_last_batch":
            engine.after_frames = lambda: skip.queue_action("quit")
        else:
            fakes.on_finish = lambda: skip.queue_action("quit")

        outcome, events, _engine = _run(tmp_path, fakes, engine, skip=skip)

        assert outcome is StageOutcome.skipped
        assert _types(events)[-1] == "batch_cancelled"
        assert _leftovers(tmp_path) == []


class TestFailing:
    def test_an_engine_error_fails_the_job_with_a_one_line_message_and_leaves_nothing(self, tmp_path, fakes):
        engine = _FakeEngine()
        engine.raise_at = 2

        outcome, events, engine = _run(tmp_path, fakes, engine)

        assert outcome is StageOutcome.failed
        failed = events[-1]
        assert failed["type"] == "batch_failed"
        assert failed["message"] == "engine blew up"
        assert next(event for event in events if event["type"] == "generation_finished")["status"] == "failed"
        assert engine.finished == 1 and fakes.reader_closed
        assert _leftovers(tmp_path) == []

    @pytest.mark.parametrize(("stops_after", "written"), [(None, 7), (2, 2)], ids=["fails after the last frame", "stops early"])
    def test_an_encoder_failure_closes_the_engine_so_it_frees_its_models_and_leaves_nothing(self, tmp_path, fakes, stops_after, written):
        fakes.writer_stops_after = stops_after
        fakes.writer_error = "ffmpeg exited: Invalid data"

        outcome, events, engine = _run(tmp_path, fakes)

        assert outcome is StageOutcome.failed
        assert "Invalid data" in events[-1]["message"]
        assert len(fakes.written) == written
        assert engine.closed and engine.finished == 1 and fakes.reader_closed
        assert _leftovers(tmp_path) == []

    @pytest.mark.parametrize("has_audio", [True, False])
    def test_a_failing_finish_step_fails_the_job_with_ffmpegs_own_error_not_its_command_line(self, tmp_path, fakes, has_audio):
        fakes.ffmpeg_fail_on = _FINISHED

        outcome, events, _engine = _run(tmp_path, fakes, request=_request(tmp_path, has_audio=has_audio))

        assert outcome is StageOutcome.failed
        assert events[-1]["type"] == "batch_failed"
        assert "boom" in events[-1]["message"]
        assert "-metadata" not in events[-1]["message"]
        assert _leftovers(tmp_path) == []

    def test_an_error_without_text_still_gives_a_message(self, tmp_path, fakes):
        engine = MagicMock()
        engine.upscale_video.side_effect = ValueError()

        _outcome, events, _engine = _run(tmp_path, fakes, engine)

        assert events[-1]["message"] == "ValueError"


class TestTimeLeft:
    def test_each_batch_reports_the_time_left_and_the_first_event_none(self, tmp_path, fakes):
        clock = _Clock()
        engine = _FakeEngine(clock, seconds_per_batch=10.0)
        fakes.source_frames = _frames(4)
        request = _request(tmp_path, frame_count=4, batch_size=1)

        _outcome, events, _engine = _run(tmp_path, fakes, engine, request=request, clock=clock)

        steps = [event for event in events if event["type"] == "step_progress"]
        assert [(event["current_step"], event["total_steps"]) for event in steps] == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
        assert "eta_secs" not in steps[0]
        assert [event["eta_secs"] for event in steps[1:]] == [30.0, 20.0, 10.0, 0.0]
        assert {event["phase"] for event in steps} == {"video_file_upscale"}
        assert {(event["workflow_stage_name"], event["workflow_stage_index"], event["workflow_total_stages"]) for event in steps} == {("upscale", 1, 1)}

    def test_the_total_starts_as_the_batches_the_clip_needs(self, tmp_path, fakes):
        _outcome, events, _engine = _run(tmp_path, fakes, request=_request(tmp_path, frame_count=60, batch_size=25))

        first = next(event for event in events if event["type"] == "step_progress")
        assert (first["current_step"], first["total_steps"]) == (0, 3)


class TestReadVideoUpscaleSource:
    def _inspect(self, monkeypatch, tmp_path, *, audio_ok=True, **overrides):
        probe = make_probe(**{"frame_count": 72, "duration": 3.0, "color_space": "bt709", "color_primaries": "bt709", "color_trc": "bt709", **overrides})
        problems = tuple(extend_source_problems(probe, None, audio_ok=audio_ok))
        monkeypatch.setattr(runner, "inspect_extend_source", lambda path: ExtendInspection(probe, None, problems, None if problems else object(), _RECORDED_CONFIG))
        return tmp_path / "clip.mp4", problems

    @pytest.mark.parametrize(
        ("overrides", "code"),
        [
            ({"color_trc": "smpte2084"}, "hdr"),
            ({"avg_frame_rate": "2400/101"}, "variable_frame_rate"),
            ({"field_order": "tt"}, "interlaced"),
            ({"sample_aspect_ratio": "4:3"}, "non_square_pixels"),
            ({"frame_count": None}, "unreadable"),
            ({"rotation": 45}, "unreadable"),
            ({"has_audio": True, "audio_ok": False}, "unreadable_audio"),
        ],
    )
    def test_a_clip_the_upscale_would_get_wrong_is_refused_with_the_reason(self, monkeypatch, tmp_path, overrides, code):
        path, problems = self._inspect(monkeypatch, tmp_path, **overrides)
        reason = next(problem.message for problem in problems if problem.code == code)

        with pytest.raises(ValueError) as refused:
            read_video_upscale_source(path)

        assert reason in str(refused.value)

    def test_a_file_without_a_video_stream_is_refused(self, monkeypatch, tmp_path):
        problem = ExtendProblem("no_video_stream", "The file has no video stream.")
        monkeypatch.setattr(runner, "inspect_extend_source", lambda path: ExtendInspection(None, None, (problem,), None))

        with pytest.raises(ValueError) as refused:
            read_video_upscale_source(tmp_path / "clip.mp4")

        assert problem.message in str(refused.value)

    @pytest.mark.parametrize(
        ("overrides", "code"),
        [
            ({"pix_fmt": "yuv420p10le"}, "high_bit_depth"),
            ({"frame_count": 10}, "too_short"),
            ({"width": 853, "coded_width": 853, "height": 480, "coded_height": 480}, "off_grid"),
            ({"width": 32, "coded_width": 32}, "too_small"),
        ],
    )
    def test_the_problems_extend_cares_about_are_ignored(self, monkeypatch, tmp_path, overrides, code):
        path, problems = self._inspect(monkeypatch, tmp_path, **overrides)
        assert code in {problem.code for problem in problems}

        source, config = read_video_upscale_source(path)

        assert source.path == path.resolve()
        assert config == _RECORDED_CONFIG

    def test_an_already_upscaled_clip_is_accepted(self, tmp_path, monkeypatch):
        probe = make_probe(frame_count=72, duration=3.0)
        problem = ExtendProblem("upscaled", "The video was upscaled.")
        monkeypatch.setattr(runner, "inspect_extend_source", lambda path: ExtendInspection(probe, None, (problem,), None, _RECORDED_CONFIG))

        source, config = read_video_upscale_source(tmp_path / "clip.mp4")

        assert (source.frame_count, config) == (72, _RECORDED_CONFIG)

    def test_the_source_carries_the_displayed_geometry_and_timing(self, monkeypatch, tmp_path):
        path, _problems = self._inspect(monkeypatch, tmp_path, coded_width=256, coded_height=512, rotation=90)

        source, _config = read_video_upscale_source(path)

        assert (source.width, source.height, source.fps_text, source.has_audio) == (512, 256, "24/1", True)
