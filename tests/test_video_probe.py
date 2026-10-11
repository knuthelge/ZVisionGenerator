"""Tests for reading a video's stream properties from ffprobe output."""

from __future__ import annotations

import json
import subprocess
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from _media import REAL, encode_frames, ffmpeg, first_frame, numbered_frames, rotated_copy

from zvisiongenerator.utils import video_probe
from zvisiongenerator.utils.video_extend import extend_source_problems
from zvisiongenerator.utils.video_probe import audio_decodes, frame_rates_match, parse_fraction, parse_probe, probe_video


def _video_stream(**overrides: Any) -> dict[str, Any]:
    stream = {
        "index": 0,
        "codec_name": "h264",
        "profile": "High",
        "codec_type": "video",
        "width": 512,
        "height": 256,
        "pix_fmt": "yuv420p",
        "sample_aspect_ratio": "1:1",
        "field_order": "progressive",
        "r_frame_rate": "24/1",
        "avg_frame_rate": "24/1",
        "time_base": "1/12288",
        "start_time": "0.000000",
        "bits_per_raw_sample": "8",
        "nb_frames": "72",
        "nb_read_packets": "72",
        "disposition": {"attached_pic": 0},
    }
    stream.update(overrides)
    return stream


def _audio_stream(**overrides: Any) -> dict[str, Any]:
    stream = {"index": 1, "codec_name": "aac", "codec_type": "audio", "sample_rate": "48000", "channels": 2, "start_time": "0.000000"}
    stream.update(overrides)
    return stream


def _probe_json(*streams: dict[str, Any], **fmt: Any) -> dict[str, Any]:
    return {"streams": list(streams), "format": {"start_time": "0.000000", "duration": "3.000000", **fmt}}


def _rotation(angle: float) -> dict[str, Any]:
    return {"side_data_list": [{"side_data_type": "Display Matrix", "displaymatrix": "...", "rotation": angle}]}


class TestParseProbe:
    def test_reads_a_plain_clip_with_audio(self):
        probe = parse_probe(_probe_json(_video_stream(), _audio_stream(start_time="0.021333")))

        assert (probe.coded_width, probe.coded_height, probe.width, probe.height) == (512, 256, 512, 256)
        assert (probe.codec, probe.profile, probe.pix_fmt, probe.bits_per_sample) == ("h264", "High", "yuv420p", 8)
        assert (probe.sample_aspect_ratio, probe.field_order, probe.time_base) == ("1:1", "progressive", "1/12288")
        assert (probe.r_frame_rate, probe.avg_frame_rate, probe.fps) == ("24/1", "24/1", 24.0)
        assert probe.frame_count == 72
        assert probe.rotation == 0
        assert (probe.duration, probe.video_start, probe.format_start) == (3.0, 0.0, 0.0)
        assert probe.has_audio is True

    def test_a_clip_without_audio(self):
        probe = parse_probe(_probe_json(_video_stream()))

        assert probe.has_audio is False

    def test_counted_packets_win_over_the_container_frame_count(self):
        probe = parse_probe(_probe_json(_video_stream(nb_frames="100", nb_read_packets="72")))

        assert probe.frame_count == 72

    @pytest.mark.parametrize(("packets", "frames", "expected"), [(None, "72", 72), (None, "N/A", None), ("N/A", None, None), ("0", "72", 72)])
    def test_frame_count_falls_back_to_the_container_count(self, packets, frames, expected):
        probe = parse_probe(_probe_json(_video_stream(nb_read_packets=packets, nb_frames=frames)))

        assert probe.frame_count == expected

    @pytest.mark.parametrize(("angle", "expected"), [(-90, 90), (90, 270), (180, 180), (-180, 180), (-270, 270), (0, 0), (-90.0, 90)])
    def test_display_matrix_rotation_is_the_quarter_turns_a_player_applies(self, angle, expected):
        probe = parse_probe(_probe_json(_video_stream(**_rotation(angle))))

        assert probe.rotation == expected

    def test_a_clip_rotated_a_quarter_turn_reports_its_displayed_size(self):
        probe = parse_probe(_probe_json(_video_stream(width=512, height=256, **_rotation(-90))))

        assert (probe.coded_width, probe.coded_height) == (512, 256)
        assert (probe.width, probe.height, probe.rotation) == (256, 512, 90)

    def test_a_clip_rotated_the_other_way_reports_its_displayed_size(self):
        probe = parse_probe(_probe_json(_video_stream(width=512, height=256, **_rotation(90))))

        assert (probe.width, probe.height, probe.rotation) == (256, 512, 270)

    def test_a_half_turn_keeps_the_size(self):
        probe = parse_probe(_probe_json(_video_stream(**_rotation(180))))

        assert (probe.width, probe.height, probe.rotation) == (512, 256, 180)

    def test_legacy_rotate_tag_is_clockwise_degrees(self):
        probe = parse_probe(_probe_json(_video_stream(tags={"rotate": "90"})))

        assert (probe.width, probe.height, probe.rotation) == (256, 512, 90)

    def test_an_unsupported_angle_is_kept_for_the_checks(self):
        probe = parse_probe(_probe_json(_video_stream(**_rotation(-45))))

        assert probe.rotation == 45

    def test_colour_tags_are_read(self):
        probe = parse_probe(_probe_json(_video_stream(color_space="bt709", color_primaries="bt709", color_transfer="bt709")))

        assert (probe.color_space, probe.color_primaries, probe.color_trc) == ("bt709", "bt709", "bt709")

    def test_missing_and_unknown_values_are_none(self):
        probe = parse_probe(_probe_json(_video_stream(color_space="unknown", sample_aspect_ratio="N/A", field_order="N/A", profile=None)))

        assert (probe.color_space, probe.color_primaries, probe.color_trc) == (None, None, None)
        assert (probe.sample_aspect_ratio, probe.field_order, probe.profile) == (None, None, None)

    def test_cover_art_is_not_the_video_stream(self):
        cover = _video_stream(index=0, width=600, height=600, disposition={"attached_pic": 1})
        probe = parse_probe(_probe_json(cover, _video_stream(index=1), _audio_stream(index=2)))

        assert (probe.width, probe.height) == (512, 256)

    def test_stream_and_format_start_times_are_kept_apart(self):
        probe = parse_probe(_probe_json(_video_stream(start_time="0.066667"), _audio_stream(start_time="0.0"), start_time="0.066667"))

        assert (probe.video_start, probe.format_start) == (0.066667, 0.066667)

    @pytest.mark.parametrize("data", [{}, {"streams": []}, _probe_json(_audio_stream()), _probe_json(_video_stream(disposition={"attached_pic": 1}))])
    def test_a_file_without_a_video_stream_is_rejected(self, data):
        with pytest.raises(ValueError, match="no video stream"):
            parse_probe(data)

    def test_a_video_stream_without_a_size_is_rejected(self):
        with pytest.raises(ValueError, match="no size"):
            parse_probe(_probe_json({"codec_type": "video"}))


class TestFps:
    def test_fps_is_the_base_frame_rate(self):
        assert parse_probe(_probe_json(_video_stream(r_frame_rate="30000/1001", avg_frame_rate="30000/1001"))).fps == pytest.approx(29.97, abs=0.001)

    def test_an_unknown_frame_rate_is_zero(self):
        assert parse_probe(_probe_json(_video_stream(r_frame_rate="0/0"))).fps == 0.0

    @pytest.mark.parametrize(("text", "expected"), [("24/1", 24), ("25", 25), ("0/0", None), ("N/A", None), (None, None), ("abc", None)])
    def test_parse_fraction(self, text, expected):
        result = parse_fraction(text)

        assert (result is None) if expected is None else (result == expected)


class TestSquarePixels:
    @pytest.mark.parametrize(("aspect", "expected"), [(None, True), ("1:1", True), ("0:1", True), ("2:2", True), ("4:3", False), ("1:2", False)])
    def test_only_a_sample_aspect_ratio_other_than_one_is_not_square(self, aspect, expected):
        probe = parse_probe(_probe_json(_video_stream(sample_aspect_ratio=aspect)))

        assert probe.square_pixels is expected


class TestFrameRatesMatch:
    @pytest.mark.parametrize(
        ("rate", "average", "expected"),
        [
            ("24/1", "24/1", True),
            ("30000/1001", "1440000/48047", True),  # a measured average a hair off the nominal rate
            ("30000/1001", "30000/1001", True),
            ("24/1", "2400/101", False),  # variable frame rate: about 1% off
            ("30/1", "24/1", False),
            ("24/1", "0/0", True),
            ("24/1", None, True),
            ("24/1", "0/1", True),
            ("0/0", "24/1", False),
            ("0/0", "0/0", False),
            ("", "24/1", False),
        ],
    )
    def test_matches_within_a_tenth_of_a_percent(self, rate, average, expected):
        assert frame_rates_match(rate, average) is expected

    def test_tolerance_is_configurable(self):
        assert frame_rates_match("24/1", "2400/101", tolerance=0.02) is True


class TestProbeVideo:
    def test_runs_ffprobe_with_packet_counts_and_parses_its_json(self, monkeypatch):
        run = MagicMock(return_value=MagicMock(stdout=json.dumps(_probe_json(_video_stream(), _audio_stream()))))
        monkeypatch.setattr(video_probe.subprocess, "run", run)

        probe = probe_video("/clips/clip.mp4")

        command = run.call_args.args[0]
        assert command[0] == "ffprobe"
        assert {"-count_packets", "-show_streams", "-show_format"} <= set(command)
        assert command[-1] == "/clips/clip.mp4"
        assert probe.frame_count == 72

    def test_an_unreadable_file_is_a_runtime_error(self, monkeypatch):
        error = subprocess.CalledProcessError(1, ["ffprobe"], stderr="Invalid data found when processing input\n")
        monkeypatch.setattr(video_probe.subprocess, "run", MagicMock(side_effect=error))

        with pytest.raises(RuntimeError, match="Invalid data found"):
            probe_video("/clips/broken.mp4")

    def test_garbled_output_is_a_runtime_error(self, monkeypatch):
        monkeypatch.setattr(video_probe.subprocess, "run", MagicMock(return_value=MagicMock(stdout="not json")))

        with pytest.raises(RuntimeError, match="unreadable output"):
            probe_video("/clips/broken.mp4")

    def test_a_file_without_video_is_a_value_error(self, monkeypatch):
        monkeypatch.setattr(video_probe.subprocess, "run", MagicMock(return_value=MagicMock(stdout=json.dumps(_probe_json(_audio_stream())))))

        with pytest.raises(ValueError, match="no video stream"):
            probe_video("/clips/song.m4a")


class TestAudioDecodes:
    def _run(self, monkeypatch, *, returncode: int = 0, stderr: str = "", stdout: bytes = b"\x01\x02") -> MagicMock:
        run = MagicMock(return_value=MagicMock(returncode=returncode, stderr=stderr, stdout=stdout))
        monkeypatch.setattr(video_probe.subprocess, "run", run)
        return run

    def test_decodes_only_the_requested_stretch(self, monkeypatch):
        run = self._run(monkeypatch)

        assert audio_decodes("/clips/clip.mp4", start=2.5, duration=1.0) is True

        command = run.call_args.args[0]
        assert command[command.index("-ss") + 1] == "2.500000"
        assert command[command.index("-t") + 1] == "1.000000"
        assert command[command.index("-map") + 1] == "0:a:0"

    @pytest.mark.parametrize(("returncode", "stdout"), [(1, b"\x01\x02"), (1, b""), (0, b"")])
    def test_a_failed_decode_or_no_samples_fails_the_check(self, monkeypatch, returncode, stdout):
        self._run(monkeypatch, returncode=returncode, stdout=stdout)

        assert audio_decodes("/clips/clip.mp4", start=0.0, duration=1.0) is False

    def test_a_warning_with_a_clean_exit_is_readable(self, monkeypatch):
        self._run(monkeypatch, returncode=0, stderr="[aac @ 0x1] Number of scalefactor bands in group exceeds limit\n")

        assert audio_decodes("/clips/clip.mp4", start=0.0, duration=1.0) is True

    def test_a_negative_start_is_clamped(self, monkeypatch):
        run = self._run(monkeypatch)

        audio_decodes("/clips/clip.mp4", start=-1.0, duration=1.0)

        command = run.call_args.args[0]
        assert command[command.index("-ss") + 1] == "0.000000"


def _patch_corner(path, width: int, height: int) -> str:
    """Return which corner of the displayed first frame holds the red patch."""
    frame = first_frame(path, width, height)
    rows, cols = np.nonzero((frame[:, :, 0] > 200) & (frame[:, :, 1] < 60))
    return ("top" if rows.mean() < height / 2 else "bottom") + "-" + ("left" if cols.mean() < width / 2 else "right")


@REAL
class TestRealClips:
    """Probe clips made with the installed ffmpeg, so the conventions hold against the real tools."""

    @pytest.fixture
    def base(self, tmp_path):
        # 256x128 clip with a red patch in the top-left corner, lossless so the patch stays exact.
        path = tmp_path / "base.mp4"
        encode_frames(numbered_frames(30, 256, 128), path, pix_fmt="yuv444p")
        return path

    def test_an_unrotated_clip(self, base):
        probe = probe_video(base)

        assert (probe.width, probe.height, probe.rotation) == (256, 128, 0)
        assert probe.frame_count == 30
        assert probe.fps == 24.0
        assert probe.field_order == "progressive"

    @pytest.mark.parametrize(
        ("angle", "rotation", "size", "corner"),
        [("-90", 90, (128, 256), "top-right"), ("90", 270, (128, 256), "bottom-left"), ("180", 180, (256, 128), "bottom-right")],
    )
    def test_rotation_is_the_clockwise_turn_the_decoder_applies(self, base, tmp_path, angle, rotation, size, corner):
        path = rotated_copy(base, tmp_path / "rotated.mp4", angle)

        probe = probe_video(path)

        assert (probe.coded_width, probe.coded_height) == (256, 128)
        assert (probe.width, probe.height, probe.rotation) == (*size, rotation)
        assert _patch_corner(path, *size) == corner

    def test_a_rotated_clip_is_not_rejected(self, base, tmp_path):
        assert extend_source_problems(probe_video(rotated_copy(base, tmp_path / "rotated.mp4", "-90")), None) == []

    def test_an_app_made_clip_is_clean(self, tmp_path):
        # What the backend writes: raw frames through libx264 at crf 18 in yuv420p, with no colour tags.
        app = tmp_path / "app.mp4"
        ffmpeg("-f", "lavfi", "-i", "testsrc=size=256x128:rate=24", "-frames:v", "40", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(app))

        probe = probe_video(app)

        assert probe.frame_count == 40
        assert extend_source_problems(probe, None) == []

    def test_a_clip_without_a_field_order_is_not_interlaced(self, tmp_path):
        clip = tmp_path / "mpeg4.mp4"
        ffmpeg("-f", "lavfi", "-i", "testsrc=size=256x128:rate=24", "-frames:v", "30", "-c:v", "mpeg4", str(clip))

        probe = probe_video(clip)

        assert probe.field_order is None
        assert extend_source_problems(probe, None) == []

    @pytest.mark.parametrize(
        ("name", "encode", "code"),
        [
            ("interlaced", ("-pix_fmt", "yuv420p", "-x264-params", "tff=1:interlaced=1"), "interlaced"),
            ("ten_bit", ("-pix_fmt", "yuv420p10le"), "high_bit_depth"),
        ],
    )
    def test_an_unsupported_clip_is_flagged(self, tmp_path, name, encode, code):
        clip = tmp_path / f"{name}.mp4"
        ffmpeg("-f", "lavfi", "-i", "testsrc=size=256x128:rate=24", "-frames:v", "30", "-c:v", "libx264", *encode, str(clip))

        assert [p.code for p in extend_source_problems(probe_video(clip), None)] == [code]

    def test_the_audio_of_a_clip_with_sound_decodes(self, tmp_path):
        clip = tmp_path / "sound.mp4"
        ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=256x128:rate=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        )

        probe = probe_video(clip)

        assert probe.has_audio
        assert audio_decodes(clip, start=1.0, duration=1.0) is True
