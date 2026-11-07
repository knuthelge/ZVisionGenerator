"""Tests for reading the last frames of a clip."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests._factories import make_extend_source
from tests._fake_media import FakeProcess, completed, f32le, popen_serving

from zvisiongenerator.utils import video_tail
from zvisiongenerator.utils.video_extend import CARRY_FRAMES
from zvisiongenerator.utils.video_tail import (
    _build_tail_command,
    _read_tail_frames,
    _tail_start_seconds,
    build_tail_audio_command,
    read_source_tail,
    read_source_tail_audio,
    stream_frames,
)

WIDTH, HEIGHT = 8, 4
FRAME_BYTES = WIDTH * HEIGHT * 3


def _frame(value: int) -> bytes:
    return bytes([value]) * FRAME_BYTES


_popen_returning = popen_serving


class TestTailStart:
    def test_starts_the_margin_before_the_first_wanted_frame(self):
        assert _tail_start_seconds(240, 24.0) == pytest.approx((240 - 25) / 24 - 2.0)

    def test_short_clip_reads_from_the_beginning(self):
        assert _tail_start_seconds(40, 24.0) == 0.0

    def test_adds_the_offset_of_the_first_frame_in_the_file(self):
        assert _tail_start_seconds(240, 24.0, video_start=0.5) == pytest.approx((240 - 25) / 24 - 2.0 + 0.5)


class TestTailStartWithFrames:
    def test_the_wanted_frame_count_moves_the_start(self):
        assert _tail_start_seconds(240, 24.0, frames=22) == pytest.approx((240 - 22) / 24 - 2.0)


class TestTailCommand:
    def test_reads_the_first_video_stream_as_raw_rgb_without_dropping_frames(self):
        command = _build_tail_command("clip.mp4", 3.5)
        assert command[command.index("-map") + 1] == "0:v:0"
        assert command[command.index("-fps_mode") + 1] == "passthrough"
        assert command[command.index("-pix_fmt") + 1] == "rgb24"
        assert command[command.index("-f") + 1] == "rawvideo"
        assert command[-1] == "-"

    def test_seeks_before_the_input(self):
        command = _build_tail_command("clip.mp4", 3.5)
        assert command.index("-ss") < command.index("-i")
        assert command[command.index("-ss") + 1] == "3.500000"

    def test_auto_rotation_stays_on_so_a_sideways_clip_comes_out_upright(self):
        assert "-noautorotate" not in _build_tail_command("clip.mp4", 3.5)

    def test_start_zero_does_not_seek(self):
        assert "-ss" not in _build_tail_command("clip.mp4", 0.0)


class TestReadTailFrames:
    def _read(self, popen: MagicMock, **overrides: Any) -> np.ndarray:
        arguments = {"width": WIDTH, "height": HEIGHT, "fps": 24.0, "frame_count": 240}
        arguments.update(overrides)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(video_tail.subprocess, "Popen", popen)
            return _read_tail_frames("clip.mp4", **arguments)

    def test_keeps_only_the_last_frames_in_order(self):
        popen = _popen_returning(b"".join(_frame(i) for i in range(CARRY_FRAMES + 3)))
        frames = self._read(popen)
        assert frames.shape == (CARRY_FRAMES, HEIGHT, WIDTH, 3)
        assert frames.dtype == np.uint8
        assert [int(frame[0, 0, 0]) for frame in frames] == list(range(3, CARRY_FRAMES + 3))

    def test_short_read_after_a_seek_retries_from_the_beginning(self):
        popen = _popen_returning(_frame(9), b"".join(_frame(i) for i in range(CARRY_FRAMES + 3)))
        frames = self._read(popen)
        assert [int(frame[0, 0, 0]) for frame in frames] == list(range(3, CARRY_FRAMES + 3))
        first, second = (call.args[0] for call in popen.call_args_list)
        assert "-ss" in first
        assert "-ss" not in second

    def test_short_read_from_the_beginning_raises(self):
        popen = _popen_returning(_frame(1) + _frame(2))
        with pytest.raises(RuntimeError, match="only 2"):
            self._read(popen, frame_count=10)
        assert popen.call_count == 1

    def test_both_reads_short_raises(self):
        popen = _popen_returning(_frame(1), _frame(2))
        with pytest.raises(RuntimeError, match=f"last {CARRY_FRAMES} frames"):
            self._read(popen)
        assert popen.call_count == 2

    def test_partial_frame_raises(self):
        popen = _popen_returning(_frame(1) * 3 + b"\x00" * 5)
        with pytest.raises(RuntimeError, match="partial frame"):
            self._read(popen, frame_count=10)

    def test_failing_ffmpeg_raises_with_its_message(self):
        popen = _popen_returning(b"", returncode=1, stderr=b"Invalid data found")
        with pytest.raises(RuntimeError, match="Invalid data found"):
            self._read(popen, frame_count=10)

    def test_source_wrapper_passes_the_displayed_geometry(self):
        source = make_extend_source(width=WIDTH, height=HEIGHT, frame_count=10, has_audio=False, video_start=0.25)
        popen = _popen_returning(b"".join(_frame(i) for i in range(CARRY_FRAMES)))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(video_tail.subprocess, "Popen", popen)
            frames = read_source_tail(source)
        assert frames.shape == (CARRY_FRAMES, HEIGHT, WIDTH, 3)

    @pytest.mark.parametrize(("video_start", "frame_count"), [(0.0, 240), (0.5, 240), (0.5, 247), (0.0, 5240)])
    def test_the_seek_follows_where_the_clips_video_starts_in_its_file(self, video_start, frame_count):
        source = make_extend_source(width=WIDTH, height=HEIGHT, frame_count=frame_count, has_audio=False, video_start=video_start)
        popen = _popen_returning(b"".join(_frame(i) for i in range(CARRY_FRAMES)))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(video_tail.subprocess, "Popen", popen)
            read_source_tail(source)

        command = popen.call_args.args[0]
        assert float(command[command.index("-ss") + 1]) == pytest.approx(_tail_start_seconds(frame_count, 24.0, video_start))

    def test_a_frame_count_that_is_too_high_retries_from_the_beginning(self):
        # A container that overcounts puts the seek past the end, so the first read finds nothing.
        source = make_extend_source(width=WIDTH, height=HEIGHT, frame_count=5200, has_audio=False)
        popen = _popen_returning(b"", b"".join(_frame(i) for i in range(CARRY_FRAMES + 2)))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(video_tail.subprocess, "Popen", popen)
            frames = read_source_tail(source)

        assert [int(frame[0, 0, 0]) for frame in frames] == list(range(2, CARRY_FRAMES + 2))
        first, second = (call.args[0] for call in popen.call_args_list)
        assert "-ss" in first and "-ss" not in second


class TestStreamFrames:
    def _stream(self, popen: MagicMock):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(video_tail.subprocess, "Popen", popen)
            return list(stream_frames("clip.mp4", width=WIDTH, height=HEIGHT))

    def test_yields_every_frame_in_order_as_an_rgb_array(self):
        popen = _popen_returning(b"".join(_frame(i) for i in range(5)))

        frames = self._stream(popen)

        assert [int(frame[0, 0, 0]) for frame in frames] == list(range(5))
        assert all(frame.shape == (HEIGHT, WIDTH, 3) and frame.dtype == np.uint8 for frame in frames)

    def test_reads_the_whole_clip_from_the_start_without_dropping_frames(self):
        popen = _popen_returning(_frame(1))

        self._stream(popen)

        command = popen.call_args.args[0]
        assert "-ss" not in command
        assert command[command.index("-fps_mode") + 1] == "passthrough"

    def test_a_failing_ffmpeg_raises_with_its_message(self):
        popen = _popen_returning(_frame(1), returncode=1, stderr=b"Invalid data found")

        with pytest.raises(RuntimeError, match="Invalid data found"):
            self._stream(popen)

    def test_closing_the_iterator_stops_ffmpeg(self):
        processes: list[FakeProcess] = []

        def popen(command, **kwargs):
            processes.append(FakeProcess(b"".join(_frame(i) for i in range(5)), returncode=None))
            return processes[0]

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(video_tail.subprocess, "Popen", popen)
            frames = stream_frames("clip.mp4", width=WIDTH, height=HEIGHT)
            next(frames)
            frames.close()

        assert processes[0].returncode == -9


class TestReadTailFramesCount:
    """The ``frames`` argument sets how many frames at the end are kept."""

    def test_keeps_the_last_22_frames_in_order(self):
        popen = _popen_returning(b"".join(_frame(i) for i in range(30)))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(video_tail.subprocess, "Popen", popen)
            frames = _read_tail_frames("clip.mp4", width=WIDTH, height=HEIGHT, fps=24.0, frame_count=240, frames=22)
        assert frames.shape == (22, HEIGHT, WIDTH, 3)
        assert [int(frame[0, 0, 0]) for frame in frames] == list(range(8, 30))

    def test_a_short_read_names_the_wanted_count(self):
        popen = _popen_returning(_frame(1) * 3)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(video_tail.subprocess, "Popen", popen)
            with pytest.raises(RuntimeError, match="last 22 frames"):
                _read_tail_frames("clip.mp4", width=WIDTH, height=HEIGHT, fps=24.0, frame_count=10, frames=22)

    def test_the_source_wrapper_passes_the_count_on(self):
        source = make_extend_source(width=WIDTH, height=HEIGHT, frame_count=10, has_audio=False)
        popen = _popen_returning(b"".join(_frame(i) for i in range(22)))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(video_tail.subprocess, "Popen", popen)
            assert read_source_tail(source, frames=22).shape == (22, HEIGHT, WIDTH, 3)


class TestTailAudioCommand:
    def test_reads_the_sound_under_the_last_frames_as_stereo_float_at_the_rate(self):
        command = build_tail_audio_command(make_extend_source(frame_count=49, fps=24.0, video_start=0.5), 16000)
        assert command[command.index("-ss") + 1] == f"{0.5 + 24 / 24:.6f}"
        assert command.index("-ss") < command.index("-i")
        assert command[command.index("-t") + 1] == f"{25 / 24:.6f}"
        assert (command[command.index("-map") + 1], command[command.index("-f") + 1]) == ("0:a:0", "f32le")
        assert (command[command.index("-ac") + 1], command[command.index("-ar") + 1]) == ("2", "16000")


class TestTailAudioCommandWithFrames:
    def test_the_shorter_tail_starts_later_and_lasts_less(self):
        command = build_tail_audio_command(make_extend_source(frame_count=49, fps=24.0, video_start=0.5), 32000, frames=22)
        assert command[command.index("-ss") + 1] == f"{0.5 + 27 / 24:.6f}"
        assert command[command.index("-t") + 1] == f"{22 / 24:.6f}"
        assert command[command.index("-ar") + 1] == "32000"


class TestReadSourceTailAudio:
    def _read(self, monkeypatch, result, **kwargs):
        run = MagicMock(return_value=result)
        monkeypatch.setattr(video_tail.subprocess, "run", run)
        samples = read_source_tail_audio(make_extend_source(frame_count=49, fps=24.0, video_start=0.5), 16000, **kwargs)
        return samples, run

    def test_returns_the_samples_ffmpeg_printed_as_stereo_float32(self, monkeypatch):
        audio = np.stack([np.linspace(-1, 1, 100), np.linspace(1, -1, 100)]).astype(np.float32)

        samples, run = self._read(monkeypatch, completed(f32le(audio)))

        assert samples.dtype == np.float32
        assert np.array_equal(samples, audio)
        assert run.call_args.args[0] == build_tail_audio_command(make_extend_source(frame_count=49, fps=24.0, video_start=0.5), 16000)

    def test_a_partial_trailing_sample_is_dropped(self, monkeypatch):
        audio = np.ones((2, 10), dtype=np.float32)

        samples, _ = self._read(monkeypatch, completed(f32le(audio) + b"\x00\x01\x02"))

        assert samples.shape == (2, 10)

    def test_the_wanted_frame_count_is_passed_on(self, monkeypatch):
        _, run = self._read(monkeypatch, completed(f32le(np.ones((2, 4), dtype=np.float32))), frames=22)

        assert run.call_args.args[0][run.call_args.args[0].index("-t") + 1] == f"{22 / 24:.6f}"

    def test_a_failing_ffmpeg_raises_with_its_message(self, monkeypatch):
        with pytest.raises(RuntimeError, match="could not read the video's audio.*Stream map '0:a:0' matches no streams"):
            self._read(monkeypatch, completed(returncode=1, stderr=b"Stream map '0:a:0' matches no streams"))

    def test_a_clip_without_sound_at_its_end_raises(self, monkeypatch):
        with pytest.raises(RuntimeError, match="Could not read the audio"):
            self._read(monkeypatch, completed(b""))
