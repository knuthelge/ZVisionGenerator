"""Tests for turning a time in a clip into a frame, and a frame into the time ffmpeg seeks to."""

from __future__ import annotations

import json

import pytest

from tests._factories import make_probe
from tests._paths import FIXTURES
from zvisiongenerator.utils.video_point import frame_at, frame_seek_seconds, frame_total


@pytest.mark.parametrize(
    ("seconds", "start", "fps", "frames", "expected"),
    [
        (0.0, 0.0, 24.0, 73, 0),
        (0.04, 0.0, 24.0, 73, 0),
        (1 / 24, 0.0, 24.0, 73, 1),
        (1.0, 0.0, 24.0, 73, 24),
        (3.2, 0.0, 25.0, 200, 80),
        (3.2 - 0.00001, 0.0, 25.0, 200, 80),
        (3.2 - 0.03, 0.0, 25.0, 200, 79),
        (0.5, 0.25, 24.0, 73, 6),
        (0.1, 0.25, 24.0, 73, 0),
        (-5.0, 0.0, 24.0, 73, 0),
        (99.0, 0.0, 24.0, 73, 72),
        (2.5, 0.0, 30000 / 1001, 100, 74),
        (7.0, 0.0, 24.0, 1, 0),
    ],
)
def test_a_time_lies_in_the_frame_that_is_showing_then(seconds, start, fps, frames, expected):
    assert frame_at(seconds, start=start, fps=fps, frames=frames) == expected


# The same cases run against ``frameAt`` in ``frontend/src/lib/state/framePoint.test.ts``: the player and the server must pick one frame.
FRAME_POINT_CASES = json.loads((FIXTURES / "frame_point_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", FRAME_POINT_CASES, ids=[case["name"] for case in FRAME_POINT_CASES])
def test_the_frame_matches_the_one_the_player_picks(case):
    assert frame_at(case["seconds"], start=case["start"], fps=case["fps"], frames=case["frames"]) == case["frame"]


def test_a_time_made_from_a_frame_lands_on_that_frame():
    fps, start = 30000 / 1001, 0.021
    assert all(frame_at(start + frame / fps, start=start, fps=fps, frames=300) == frame for frame in range(300))


@pytest.mark.parametrize(("fps", "frames"), [(0.0, 10), (-24.0, 10), (24.0, 0)])
def test_a_clip_without_a_rate_or_frames_has_no_frame(fps, frames):
    with pytest.raises(ValueError, match="frame rate"):
        frame_at(1.0, start=0.0, fps=fps, frames=frames)


@pytest.mark.parametrize(("frame", "start", "expected"), [(0, 0.0, 0.0), (1, 0.0, 0.5 / 24), (24, 0.0, 23.5 / 24), (24, 0.5, 0.5 + 23.5 / 24), (0, 0.5, 0.5 - 0.5 / 24)])
def test_the_seek_is_half_a_frame_before_the_frame(frame, start, expected):
    assert frame_seek_seconds(frame, start=start, fps=24.0) == pytest.approx(expected)


def test_the_seek_of_the_first_frame_is_never_negative():
    assert frame_seek_seconds(0, start=0.0, fps=24.0) == 0.0


def test_the_seek_target_decodes_the_frame_it_was_made_for():
    fps, start = 25.0, 0.04
    seek = frame_seek_seconds(7, start=start, fps=fps)

    assert start + 6 / fps < seek < start + 7 / fps


def test_the_frame_count_is_the_counted_one_else_the_length_times_the_rate():
    assert frame_total(make_probe(frame_count=73)) == 73
    assert frame_total(make_probe(frame_count=None, duration=4.0, r_frame_rate="25/1")) == 100


@pytest.mark.parametrize("overrides", [{"frame_count": None, "duration": None}, {"frame_count": None, "r_frame_rate": "0/0"}])
def test_a_clip_with_no_way_to_count_its_frames_has_none(overrides):
    assert frame_total(make_probe(**overrides)) is None
