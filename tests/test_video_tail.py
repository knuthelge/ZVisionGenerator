"""Tests for reading the last frames of a clip."""

from __future__ import annotations

import dataclasses
import io
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from _media import REAL, decode_frames, encode_frames, ffmpeg, numbered_frames, rotated_copy
from conftest import make_extend_source
from zvisiongenerator.core.video_types import ExtendSource
from zvisiongenerator.utils import video_tail
from zvisiongenerator.utils.video_extend import build_extend_source
from zvisiongenerator.utils.video_probe import probe_video
from zvisiongenerator.utils.video_extend import CARRY_FRAMES
from zvisiongenerator.utils.video_tail import _build_tail_command, _read_tail_frames, _tail_start_seconds, build_tail_audio_command, read_source_tail, read_source_tail_audio

WIDTH, HEIGHT = 8, 4
FRAME_BYTES = WIDTH * HEIGHT * 3


def _frame(value: int) -> bytes:
    return bytes([value]) * FRAME_BYTES


class _FakeProcess:
    """A stand-in for ``subprocess.Popen`` whose stdout is the given bytes."""

    def __init__(self, output: bytes, returncode: int = 0, stderr_text: bytes = b"", stderr_file: Any = None) -> None:
        self.stdout = io.BytesIO(output)
        self.returncode = returncode
        if stderr_file is not None:
            stderr_file.write(stderr_text)

    def poll(self) -> int:
        return self.returncode

    def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def _popen_returning(*outputs: bytes, returncode: int = 0, stderr_text: bytes = b"") -> MagicMock:
    """Return a ``Popen`` replacement that serves each output in turn and records the commands."""
    queue = list(outputs)

    def popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        return _FakeProcess(queue.pop(0), returncode, stderr_text, kwargs.get("stderr"))

    return MagicMock(side_effect=popen)


class TestTailStart:
    def test_starts_the_margin_before_the_first_wanted_frame(self):
        assert _tail_start_seconds(240, 24.0) == pytest.approx((240 - 25) / 24 - 2.0)

    def test_short_clip_reads_from_the_beginning(self):
        assert _tail_start_seconds(40, 24.0) == 0.0

    def test_adds_the_offset_of_the_first_frame_in_the_file(self):
        assert _tail_start_seconds(240, 24.0, video_start=0.5) == pytest.approx((240 - 25) / 24 - 2.0 + 0.5)


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

    def test_start_zero_does_not_seek(self):
        assert "-ss" not in _build_tail_command("clip.mp4", 0.0)

    def test_keeps_auto_rotation(self):
        assert "-noautorotate" not in _build_tail_command("clip.mp4", 1.0)


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

    def test_returned_array_is_writable(self):
        frames = self._read(_popen_returning(b"".join(_frame(i) for i in range(CARRY_FRAMES))))
        frames[0, 0, 0, 0] = 1

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
        popen = _popen_returning(b"", returncode=1, stderr_text=b"Invalid data found")
        with pytest.raises(RuntimeError, match="Invalid data found"):
            self._read(popen, frame_count=10)

    def test_source_wrapper_passes_the_displayed_geometry(self):
        source = make_extend_source(width=WIDTH, height=HEIGHT, frame_count=10, has_audio=False, video_start=0.25)
        popen = _popen_returning(b"".join(_frame(i) for i in range(CARRY_FRAMES)))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(video_tail.subprocess, "Popen", popen)
            frames = read_source_tail(source)
        assert frames.shape == (CARRY_FRAMES, HEIGHT, WIDTH, 3)


def _tail_of(path: Path) -> tuple[np.ndarray, ExtendSource]:
    probe = probe_video(path)
    source = build_extend_source(path, probe, None)
    return read_source_tail(source), source


@REAL
class TestRealTail:
    """Tail reads against clips made with the installed ffmpeg."""

    def test_short_clip_returns_frames_36_to_60_in_order(self, tmp_path):
        frames = numbered_frames(61)
        clip = tmp_path / "short.mp4"
        encode_frames(frames, clip, "-g", "12")
        tail, source = _tail_of(clip)
        assert source.frame_count == 61
        assert tail.shape == (25, 32, 64, 3)
        assert np.array_equal(tail, decode_frames(clip, 64, 32)[36:61])
        for offset, frame in enumerate(tail):
            assert np.abs(frame[16:, 16:].astype(int) - frames[36 + offset][16:, 16:].astype(int)).max() <= 3

    def test_long_clip_seeks_and_still_returns_the_last_frames(self, tmp_path):
        frames = numbered_frames(200)
        clip = tmp_path / "long.mp4"
        encode_frames(frames, clip, "-g", "12")
        decoded = decode_frames(clip, 64, 32)
        assert len({frame[16:, 16:].tobytes() for frame in decoded}) == 200  # every frame is told apart
        source = build_extend_source(clip, probe_video(clip), None)
        assert video_tail._tail_start_seconds(source.frame_count, source.fps) > 0  # the read really seeks
        tail, _ = _tail_of(clip)
        assert np.array_equal(tail, decoded[175:200])

    def test_clip_that_starts_late_in_its_file(self, tmp_path):
        frames = numbered_frames(200)
        raw = tmp_path / "plain.mp4"
        encode_frames(frames, raw, "-g", "12")
        clip = tmp_path / "late.mkv"
        ffmpeg("-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-itsoffset", "0.5", "-i", str(raw), "-map", "1:v", "-map", "0:a", "-c:v", "copy", "-c:a", "aac", "-t", "9", str(clip))
        probe = probe_video(clip)
        assert probe.video_start == pytest.approx(0.5)
        tail, source = _tail_of(clip)
        assert source.video_start == pytest.approx(0.5 - (probe.format_start or 0.0))  # the AAC priming may start the container early
        assert np.array_equal(tail, decode_frames(clip, 64, 32)[175:200])

    @pytest.mark.parametrize("angle", ["-90", "90", "180"])
    def test_rotated_clip_comes_out_upright_at_the_displayed_size(self, tmp_path, angle):
        frames = numbered_frames(61, width=64, height=32)
        base = tmp_path / "base.mp4"
        encode_frames(frames, base, "-g", "12")
        clip = rotated_copy(base, tmp_path / "rotated.mp4", angle)
        probe = probe_video(clip)
        tail, source = _tail_of(clip)
        assert (source.width, source.height) == (probe.width, probe.height)
        expected = decode_frames(clip, probe.width, probe.height)[36:61]  # ffmpeg's own auto-rotated decode
        assert tail.shape == (25, probe.height, probe.width, 3)
        assert np.array_equal(tail, expected)

    def test_clip_shorter_than_the_carry_raises(self, tmp_path):
        clip = tmp_path / "tiny.mp4"
        encode_frames(numbered_frames(10), clip, "-g", "12")
        with pytest.raises(RuntimeError, match="last 25 frames"):
            _tail_of(clip)


@REAL
class TestRealTailWithWrongFrameCount:
    """A frame count that is off (a container's ``nb_frames`` can be) must not make the reader return the wrong frames."""

    @pytest.mark.parametrize("error", [0, -1, 1, 7, -300, 5000])
    def test_still_returns_the_true_last_frames(self, tmp_path, error):
        clip = tmp_path / "long.mp4"
        encode_frames(numbered_frames(200), clip, "-g", "12", "-bf", "3", quality=("-crf", "12"))  # lossless x264 writes no B-frames
        packets = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts,dts", "-of", "csv=p=0", str(clip)], capture_output=True, text=True, check=True
        ).stdout
        assert any(line.split(",")[0] != line.split(",")[1] for line in packets.split())  # decode order differs from display order
        decoded = decode_frames(clip, 64, 32)
        assert len(decoded) == 200
        source = build_extend_source(clip, probe_video(clip), None)
        assert source.frame_count == 200  # the probe counts packets
        wrong = dataclasses.replace(source, frame_count=200 + error)
        assert np.array_equal(read_source_tail(wrong), decoded[175:200])

    def test_overstated_count_on_a_short_clip_raises_instead_of_padding(self, tmp_path):
        clip = tmp_path / "short.mp4"
        encode_frames(numbered_frames(20), clip, "-g", "12")
        source = build_extend_source(clip, probe_video(clip), None)
        with pytest.raises(RuntimeError, match="found only 20"):
            read_source_tail(dataclasses.replace(source, frame_count=500))


class TestTailAudioCommand:
    def test_reads_the_sound_under_the_last_frames_as_stereo_float_at_the_rate(self):
        command = build_tail_audio_command(make_extend_source(frame_count=49, fps=24.0, video_start=0.5), 16000)
        assert command[command.index("-ss") + 1] == f"{0.5 + 24 / 24:.6f}"
        assert command.index("-ss") < command.index("-i")
        assert command[command.index("-t") + 1] == f"{25 / 24:.6f}"
        assert (command[command.index("-map") + 1], command[command.index("-f") + 1]) == ("0:a:0", "f32le")
        assert (command[command.index("-ac") + 1], command[command.index("-ar") + 1]) == ("2", "16000")


@REAL
class TestReadSourceTailAudio:
    def test_returns_the_tails_sound_from_its_position_in_the_file(self, tmp_path):
        path = tmp_path / "clip.mp4"
        tone = "aevalsrc=if(gt(t\\,1)\\,0.5*sin(2*PI*440*t)\\,0):s=44100:d=2"
        raw = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "64x32", "-r", "24", "-i", "-"]
        ffmpeg("-f", "lavfi", "-i", tone, *raw, "-map", "1:v", "-map", "0:a", "-c:v", "libx264", "-c:a", "pcm_s16le", "-shortest", str(path.with_suffix(".mov")), data=numbered_frames(48).tobytes())
        source = build_extend_source(path.with_suffix(".mov"), probe_video(path.with_suffix(".mov")), None)
        samples = read_source_tail_audio(source, 16000)
        assert samples.shape[0] == 2 and samples.dtype == np.float32
        assert abs(samples.shape[1] - round(25 / 24 * 16000)) <= 16
        lead = round((1 - 23 / 24) * 16000)  # the tail starts at frame 23, before the tone starts at 1 s
        assert np.abs(samples[:, : lead - 80]).max() < 0.01
        assert np.abs(samples[:, lead + 160 :]).max() > 0.3  # the mono tone is upmixed to stereo at -3 dB

    def test_a_clip_without_sound_is_an_error(self, tmp_path):
        path = tmp_path / "clip.mp4"
        encode_frames(numbered_frames(48), path)
        source = dataclasses.replace(build_extend_source(path, probe_video(path), None), has_audio=True)
        with pytest.raises(RuntimeError):
            read_source_tail_audio(source, 16000)
