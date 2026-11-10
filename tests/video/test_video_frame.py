"""Tests for reading one frame of a clip as a PNG and saving it; ffprobe and ffmpeg are mocked."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests._factories import make_probe
from tests._fake_media import completed
from tests._paths import FIXTURES
from zvisiongenerator.utils import video_frame
from zvisiongenerator.utils.video_frame import (
    extract_frame_png,
    frame_color_filter,
    extract_last_frame_png,
    frame_file_name,
    frame_png_command,
    last_frame_png_command,
    locate_frame,
    locate_last_frame,
    save_video_frame,
)

PNG = b"\x89PNG\r\n\x1a\nfull size frame"


def _probe_json(**stream: object) -> dict:
    return {
        "streams": [{"codec_type": "video", "width": 512, "height": 256, "r_frame_rate": "25/1", "avg_frame_rate": "25/1", "nb_frames": "100", "start_time": "0.000000", **stream}],
        "format": {"duration": "4.0", "start_time": "0.000000"},
    }


@pytest.fixture
def ffmpeg(monkeypatch) -> MagicMock:
    """Mock ffprobe (a 100-frame, 25 fps clip) and the ffmpeg call; returns the ``subprocess.run`` mock."""
    monkeypatch.setattr(video_frame, "ffprobe_json", MagicMock(return_value=_probe_json()))
    run = MagicMock(return_value=completed(PNG))
    monkeypatch.setattr(subprocess, "run", run)
    return run


# ── Which frame ──────────────────────────────────────────────────────────────


def test_a_time_is_snapped_to_its_frame():
    target = locate_frame(make_probe(frame_count=100, r_frame_rate="25/1"), 3.21)

    assert target.frame == 80
    assert target.seek == pytest.approx(79.5 / 25)


def test_the_start_of_the_video_stream_moves_the_frame():
    probe = make_probe(frame_count=100, r_frame_rate="25/1", video_start=0.08, format_start=0.0)

    assert locate_frame(probe, 0.08).frame == 0
    assert locate_frame(probe, 0.12).frame == 1


def test_a_time_past_the_end_is_the_last_frame():
    assert locate_frame(make_probe(frame_count=100, r_frame_rate="25/1"), 500.0).frame == 99


def test_the_last_frame_is_the_one_the_clip_ends_on():
    target = locate_last_frame(make_probe(frame_count=100, r_frame_rate="25/1", video_start=0.08, format_start=0.0))

    assert target.frame == 99
    assert target.seek == pytest.approx(0.08 + 98.5 / 25)


@pytest.mark.parametrize("overrides", [{"r_frame_rate": "0/0"}, {"frame_count": None, "duration": None}])
def test_a_clip_with_an_unknown_rate_or_length_has_no_frame(overrides):
    with pytest.raises(ValueError, match="frame rate or length"):
        locate_frame(make_probe(**overrides), 1.0)
    with pytest.raises(ValueError, match="frame rate or length"):
        locate_last_frame(make_probe(**overrides))


# ── Colour ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("space", [None, "unknown", "unspecified"])
def test_a_clip_without_a_colour_matrix_is_converted_the_way_a_player_shows_it(space):
    assert frame_color_filter(make_probe(color_space=space)) == "scale=in_color_matrix=bt709,format=rgb24"


@pytest.mark.parametrize("space", ["bt709", "smpte170m", "bt470bg"])
def test_a_clip_with_a_colour_matrix_is_left_to_its_tag(space):
    assert frame_color_filter(make_probe(color_space=space)) is None


# ── The command ──────────────────────────────────────────────────────────────


def test_the_command_seeks_before_the_input_and_prints_one_png():
    command = frame_png_command(Path("/clips/fox.mp4"), 1.98, None)

    assert command[command.index("-ss") + 1] == "1.980000"
    assert command.index("-ss") < command.index("-i")
    assert command[command.index("-i") + 1] == str(Path("/clips/fox.mp4"))
    assert command[command.index("-frames:v") + 1] == "1"
    assert command[-3:] == ["-c:v", "png", "-"]
    assert "-vf" not in command


def test_the_command_keeps_the_colour_filter_and_shrinks_after_it():
    command = frame_png_command(Path("clip.mp4"), 0.0, "scale=in_color_matrix=bt709,format=rgb24", edge=384)

    assert command[command.index("-vf") + 1] == "scale=in_color_matrix=bt709,format=rgb24,scale=384:384:force_original_aspect_ratio=decrease"


def test_the_command_shrinks_without_a_colour_filter():
    assert frame_png_command(Path("clip.mp4"), 0.0, None, edge=384)[-8] == "scale=384:384:force_original_aspect_ratio=decrease"


def test_the_last_frame_command_seeks_from_the_end_and_rewrites_one_file():
    command = last_frame_png_command(Path("/clips/fox.webm"), Path("/scratch/last.png"), "scale=in_color_matrix=bt709,format=rgb24", edge=384)

    assert command[command.index("-sseof") + 1] == "-1"
    assert command.index("-sseof") < command.index("-i")
    assert command[command.index("-update") + 1] == "1"
    assert command[command.index("-vf") + 1] == "scale=in_color_matrix=bt709,format=rgb24,scale=384:384:force_original_aspect_ratio=decrease"
    assert command[-1] == str(Path("/scratch/last.png"))


def test_the_last_frame_by_time_is_what_ffmpeg_left_in_the_file(monkeypatch):
    seen: list[list[str]] = []

    def _run(command, **_kwargs):
        seen.append(command)
        Path(command[-1]).write_bytes(b"png-bytes")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _run)

    assert extract_last_frame_png(Path("clip.webm"), make_probe(color_space="bt709"), edge=384) == b"png-bytes"
    assert seen[0][seen[0].index("-vf") + 1] == "scale=384:384:force_original_aspect_ratio=decrease"


def test_the_last_frame_by_time_is_refused_when_ffmpeg_writes_none(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b""))

    with pytest.raises(RuntimeError, match="no frame"):
        extract_last_frame_png(Path("clip.webm"), make_probe())


# ── Names ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("clip", "seconds", "expected"),
    [
        ("fox", 3.2, "fox_3.2s.png"),
        ("fox", 0.0, "fox_0.0s.png"),
        ("a:b*c?d", 1.25, "a_b_c_d_1.3s.png"),
        ("", 2.0, "frame_2.0s.png"),
        ("clip (2)", 10.04, "clip (2)_10.0s.png"),
        ("x" * 300, 1.0, "x" * 100 + "_1.0s.png"),
    ],
)
def test_a_frame_is_named_after_its_clip_and_time(clip, seconds, expected):
    assert frame_file_name(clip, seconds) == expected


# The same cases run against ``pointLabel`` in ``frontend/src/lib/state/framePoint.test.ts``: the label and the file name must read one time.
POINT_LABEL_CASES = json.loads((FIXTURES / "point_label_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", POINT_LABEL_CASES, ids=[case["name"] for case in POINT_LABEL_CASES])
def test_a_frame_name_reads_the_time_as_the_label_does(case):
    assert frame_file_name("fox", case["seconds"]) == f"fox_{case['label'].removesuffix(' s')}s.png"


def test_a_frame_whose_time_is_a_tie_is_named_as_it_is_labelled(ffmpeg, tmp_path):
    video_frame.ffprobe_json.return_value = _probe_json(r_frame_rate="24/1", avg_frame_rate="24/1")

    saved = save_video_frame(tmp_path / "fox.mp4", 30 / 24 + 0.01, tmp_path)

    assert saved.name == "fox_1.3s.png"


# ── Saving ───────────────────────────────────────────────────────────────────


def test_a_frame_is_saved_under_the_hash_of_its_picture(ffmpeg, tmp_path):
    saved = save_video_frame(tmp_path / "fox.mp4", 3.21, tmp_path / ".web_uploads")

    assert saved == tmp_path / ".web_uploads" / hashlib.sha256(PNG).hexdigest() / "fox_3.2s.png"
    assert saved.read_bytes() == PNG
    command = ffmpeg.call_args.args[0]
    assert command[command.index("-ss") + 1] == "3.180000"
    assert ffmpeg.call_args.kwargs["timeout"] == video_frame.FRAME_TIMEOUT


def test_the_same_picture_is_not_written_again(ffmpeg, tmp_path):
    first = save_video_frame(tmp_path / "fox.mp4", 3.2, tmp_path)
    stamp = first.stat().st_mtime_ns

    again = save_video_frame(tmp_path / "fox.mp4", 3.2, tmp_path)

    assert again == first
    assert first.stat().st_mtime_ns == stamp


def test_another_picture_is_saved_beside_it(ffmpeg, tmp_path):
    first = save_video_frame(tmp_path / "fox.mp4", 3.2, tmp_path)
    ffmpeg.return_value = completed(PNG + b"other")

    second = save_video_frame(tmp_path / "fox.mp4", 3.2, tmp_path)

    assert second != first
    assert first.is_file()


def test_the_untagged_clip_is_converted_as_the_player_shows_it(ffmpeg, tmp_path):
    save_video_frame(tmp_path / "fox.mp4", 1.0, tmp_path)

    command = ffmpeg.call_args.args[0]
    assert command[command.index("-vf") + 1] == "scale=in_color_matrix=bt709,format=rgb24"


def test_a_tagged_clip_is_not_given_a_colour_filter(ffmpeg, tmp_path):
    video_frame.ffprobe_json.return_value = _probe_json(color_space="bt709")

    save_video_frame(tmp_path / "fox.mp4", 1.0, tmp_path)

    assert "-vf" not in ffmpeg.call_args.args[0]


def test_ffmpeg_failing_says_why_and_saves_nothing(ffmpeg, tmp_path):
    ffmpeg.return_value = completed(b"", stderr=b"Invalid data found when processing input", returncode=1)

    with pytest.raises(RuntimeError, match="could not read the frame: Invalid data found"):
        save_video_frame(tmp_path / "fox.mp4", 1.0, tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_ffmpeg_printing_nothing_is_an_error(ffmpeg, tmp_path):
    ffmpeg.return_value = completed(b"")

    with pytest.raises(RuntimeError, match="no frame"):
        extract_frame_png(tmp_path / "fox.mp4", make_probe(), locate_frame(make_probe(), 0.0).seek)


def test_ffmpeg_running_too_long_is_an_error(ffmpeg, tmp_path):
    ffmpeg.side_effect = subprocess.TimeoutExpired("ffmpeg", 60)

    with pytest.raises(RuntimeError, match="took longer"):
        save_video_frame(tmp_path / "fox.mp4", 1.0, tmp_path)


def test_a_clip_ffprobe_cannot_read_is_an_error(ffmpeg, tmp_path):
    video_frame.ffprobe_json.side_effect = RuntimeError("ffprobe could not read the clip")

    with pytest.raises(RuntimeError, match="could not read"):
        save_video_frame(tmp_path / "fox.mp4", 1.0, tmp_path)
    ffmpeg.assert_not_called()
