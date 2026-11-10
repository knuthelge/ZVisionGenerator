"""Tests for the contact sheet of a clip; ffprobe and ffmpeg are mocked."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests._factories import make_probe
from tests._fake_media import completed
from zvisiongenerator.utils import video_frame
from zvisiongenerator.utils.video_sprite import SPRITE_MAX_TILES, build_sprite, sprite_command, sprite_frames

JPEG = b"\xff\xd8\xff contact sheet"


@pytest.mark.parametrize("frames", [2, 10, 73, 120, 121])
def test_a_clip_of_121_frames_or_fewer_has_a_tile_for_each_frame(frames):
    assert sprite_frames(frames) == list(range(frames))


@pytest.mark.parametrize("frames", [122, 500, 2400, 100_000])
def test_a_longer_clip_has_121_evenly_spaced_tiles_from_the_first_frame_to_the_last(frames):
    picks = sprite_frames(frames)

    assert len(picks) == SPRITE_MAX_TILES
    assert picks[0] == 0
    assert picks[-1] == frames - 1
    assert picks == sorted(set(picks))
    assert picks[60] == round(60 / 120 * (frames - 1))


def test_a_one_frame_clip_is_one_tile():
    assert sprite_frames(1) == [0]


def test_a_clip_without_frames_has_no_sheet():
    with pytest.raises(ValueError, match="at least one frame"):
        sprite_frames(0)


def test_the_command_picks_the_frames_and_tiles_them_into_one_jpeg():
    command = sprite_command(Path("/clips/fox.mp4"), make_probe(width=1920, height=1080), [0, 5, 9])
    chain = command[command.index("-vf") + 1]

    assert chain.startswith("select='eq(n,0)+eq(n,5)+eq(n,9)',")
    assert chain.endswith(",scale=160:-2,tile=11x11")
    assert command[command.index("-frames:v") + 1] == "1"
    assert command[command.index("-c:v") + 1] == "mjpeg"
    assert "-an" in command
    assert command[-1] == "-"


def test_a_tall_clip_is_fitted_by_its_height():
    command = sprite_command(Path("fox.mp4"), make_probe(width=720, height=1280), [0])

    assert ",scale=-2:160,tile=11x11" in command[command.index("-vf") + 1]


def test_an_untagged_clip_is_converted_as_the_player_shows_it_before_it_shrinks():
    command = sprite_command(Path("fox.mp4"), make_probe(color_space=None), [0])
    chain = command[command.index("-vf") + 1]

    assert chain.index("scale=in_color_matrix=bt709,format=rgb24") < chain.index("scale=160:-2")


def test_a_tagged_clip_has_no_colour_filter():
    command = sprite_command(Path("fox.mp4"), make_probe(color_space="bt709"), [0])

    assert "in_color_matrix" not in command[command.index("-vf") + 1]


@pytest.fixture
def ffmpeg(monkeypatch) -> MagicMock:
    probe = {"streams": [{"codec_type": "video", "width": 512, "height": 256, "r_frame_rate": "25/1", "nb_frames": "300"}], "format": {"duration": "12.0"}}
    monkeypatch.setattr(video_frame, "ffprobe_json", MagicMock(return_value=probe))
    run = MagicMock(return_value=completed(JPEG))
    monkeypatch.setattr(subprocess, "run", run)
    return run


def test_the_sheet_is_the_jpeg_of_one_ffmpeg_call_with_the_frames_the_facts_report(ffmpeg):
    assert build_sprite(Path("fox.mp4")) == JPEG

    assert ffmpeg.call_count == 1
    command = ffmpeg.call_args.args[0]
    chain = command[command.index("-vf") + 1]
    assert chain.count("eq(n,") == SPRITE_MAX_TILES
    assert "+eq(n,299)'," in chain


def test_ffmpeg_failing_is_an_error_that_says_why(ffmpeg):
    ffmpeg.return_value = completed(b"", stderr=b"Invalid data found", returncode=1)

    with pytest.raises(RuntimeError, match="could not build the contact sheet: Invalid data found"):
        build_sprite(Path("fox.mp4"))


def test_ffmpeg_printing_nothing_is_an_error(ffmpeg):
    ffmpeg.return_value = completed(b"")

    with pytest.raises(RuntimeError, match="no contact sheet"):
        build_sprite(Path("fox.mp4"))


def test_a_clip_of_unknown_length_has_no_sheet(ffmpeg):
    video_frame.ffprobe_json.return_value = {"streams": [{"codec_type": "video", "width": 64, "height": 64}], "format": {}}

    with pytest.raises(ValueError, match="length"):
        build_sprite(Path("fox.mp4"))
    ffmpeg.assert_not_called()
