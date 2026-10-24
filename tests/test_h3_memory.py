"""Tests for the MiniMax H3 row and frame arithmetic and clip geometry."""

from __future__ import annotations

import pytest

from zvisiongenerator.backends.cuda.video.h3.frames import Geometry, geometry
from zvisiongenerator.utils import h3_memory
from zvisiongenerator.utils.h3_memory import (
    H3_TEXT_TOKENS,
    h3_audio_latents,
    h3_latent_frames,
    h3_sequence_rows,
    h3_video_rows,
)


@pytest.mark.parametrize(("frames", "latents"), [(22, 7), (39, 12), (124, 37), (243, 72)])
def test_latent_frames(frames, latents):
    assert h3_latent_frames(frames) == latents


@pytest.mark.parametrize("frames", [0, 1, 21, 23, 125, -12])
def test_latent_frames_refuses_other_counts(frames):
    with pytest.raises(ValueError, match="17 \\* n \\+ 5"):
        h3_latent_frames(frames)


@pytest.mark.parametrize(("frames", "latents"), [(22, 37), (124, 207), (243, 405)])
def test_audio_latents(frames, latents):
    assert h3_audio_latents(frames) == latents


def test_video_rows_pin():
    assert h3_video_rows(864, 480, 124) == 14_985


def test_sequence_rows_pin_adds_text_and_audio():
    assert h3_sequence_rows(864, 480, 124) == 15_499
    assert h3_sequence_rows(864, 480, 124) == 14_985 + 207 * 2 + H3_TEXT_TOKENS


@pytest.mark.parametrize(
    ("width", "height", "video", "sequence"),
    [
        (576, 320, 6_660, 7_174),
        (768, 448, 12_432, 12_946),
        (1024, 576, 21_312, 21_826),
        (1280, 704, 32_560, 33_074),
        (1600, 896, 51_800, 52_314),
        (384, 384, 5_328, 5_842),
        (576, 576, 11_988, 12_502),
        (768, 768, 21_312, 21_826),
        (960, 960, 33_300, 33_814),
        (1216, 1216, 53_428, 53_942),
    ],
)
def test_rows_at_124_frames(width, height, video, sequence):
    assert (h3_video_rows(width, height, 124), h3_sequence_rows(width, height, 124)) == (video, sequence)


def test_keyframes_add_one_latent_frame_of_rows_each():
    cell = (864 // 32) * (480 // 32)
    assert cell == 405
    assert h3_sequence_rows(864, 480, 124, keyframes=1) == 15_499 + cell
    assert h3_sequence_rows(864, 480, 124, keyframes=2) == 15_499 + 2 * cell


def test_tail_adds_no_rows():
    assert h3_sequence_rows(864, 480, 124, tail_frames=22) == h3_sequence_rows(864, 480, 124)


@pytest.mark.parametrize(
    "kwargs",
    [{"keyframes": -1}, {"tail_frames": 21}, {"tail_frames": 141}],
)
def test_sequence_rows_refuses_bad_conditioning(kwargs):
    with pytest.raises(ValueError):
        h3_sequence_rows(864, 480, 124, **kwargs)


@pytest.mark.parametrize(("width", "height"), [(864, 481), (850, 480), (0, 480), (864, 0), (-32, 480)])
def test_unaligned_canvas_is_refused(width, height):
    with pytest.raises(ValueError, match="multiple of 32"):
        h3_video_rows(width, height, 124)
    with pytest.raises(ValueError, match="multiple of 32"):
        geometry(width, height, 124)


def test_geometry_pin():
    geo = geometry(864, 480, 124)
    assert geo == Geometry(864, 480, 124, latent_frames=37, latent_height=30, latent_width=54, audio_latents=207)
    assert geo.video_rows == 14_985
    assert geo.audio_rows == 414


@pytest.mark.parametrize(("width", "height", "frames"), [(864, 480, 124), (768, 448, 22), (1280, 736, 243), (64, 32, 5)])
def test_geometry_agrees_with_the_arithmetic(width, height, frames):
    geo = geometry(width, height, frames)
    assert geo.video_rows == h3_video_rows(width, height, frames)
    assert geo.audio_rows + geo.video_rows + H3_TEXT_TOKENS == h3_sequence_rows(width, height, frames)


def test_geometry_refuses_a_bad_frame_count():
    with pytest.raises(ValueError, match="17 \\* n \\+ 5"):
        geometry(864, 480, 120)


def test_geometry_is_frozen():
    with pytest.raises(AttributeError):
        geometry(864, 480, 124).width = 1  # type: ignore[misc]


def test_public_names_are_exported():
    assert all(hasattr(h3_memory, name) for name in h3_memory.__all__)
