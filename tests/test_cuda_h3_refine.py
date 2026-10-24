"""Tests for the H3 refine helpers: noise, audio windows, window keyframes and the cross-fade, on tiny tensors."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zvisiongenerator.backends.cuda.video.h3.refine import (  # noqa: E402
    audio_window,
    draw_refine_noise,
    stitch,
    window_audio_rows,
    window_conditioning,
)
from zvisiongenerator.utils.h3_memory import H3_REFINE_OVERLAP_LATENT_FRAMES, h3_refine_windows  # noqa: E402

_FIRST = torch.full((1, 24, 1, 2, 2), 1.0)
_LAST = torch.full((1, 24, 1, 2, 2), 2.0)


def test_the_noise_is_deterministic_per_seed_and_differs_between_seeds() -> None:
    video, audio = draw_refine_noise((1, 24, 7, 2, 2), (2, 32, 11), 5)
    again_video, again_audio = draw_refine_noise((1, 24, 7, 2, 2), (2, 32, 11), 5)
    other_video, other_audio = draw_refine_noise((1, 24, 7, 2, 2), (2, 32, 11), 6)
    assert video.shape == (1, 24, 7, 2, 2)
    assert audio.shape == (2, 32, 11)
    assert video.dtype == audio.dtype == torch.float32
    assert torch.equal(video, again_video)
    assert torch.equal(audio, again_audio)
    assert not torch.equal(video, other_video)
    assert not torch.equal(audio, other_audio)


def test_the_noise_is_the_video_then_the_audio_of_one_generator_seeded_one_above_the_seed() -> None:
    video, audio = draw_refine_noise((1, 24, 7, 2, 2), (2, 32, 11), 5)
    generator = torch.Generator().manual_seed(6)
    assert torch.equal(video, torch.randn(1, 24, 7, 2, 2, generator=generator))
    assert torch.equal(audio, torch.randn(2, 32, 11, generator=generator))


def test_a_single_window_takes_all_the_audio() -> None:
    assert audio_window(207, 0, 37, 124) == (0, 207)


def test_chunk_audio_offsets_are_proportional_and_stay_inside_the_clip() -> None:
    windows = h3_refine_windows(72)
    audio_latents = 40 * 243 // 24
    count = 122  # h3_audio_latents(73)
    offsets = [audio_window(audio_latents, start, 72, 73) for start, _end in windows]
    assert all(window_count == count for _offset, window_count in offsets)
    assert [offset for offset, _count in offsets] == sorted(offset for offset, _count in offsets)
    assert all(offset + window_count <= audio_latents for offset, window_count in offsets)
    assert offsets[0][0] == 0
    assert offsets[1][0] == round(15 / 72 * audio_latents)


def test_an_offset_that_would_run_past_the_end_is_clamped() -> None:
    assert audio_window(130, 30, 32, 73) == (130 - 122, 122)


def test_window_rows_are_channel_major_with_the_same_positions_in_both_channels() -> None:
    n = 9
    audio = torch.zeros(2, 32, n)
    for channel in range(2):
        for position in range(n):
            audio[channel, :, position] = 100 * channel + position
    rows = window_audio_rows(audio, 3, 4)
    assert rows.shape == (8, 32)
    assert rows[:4, 0].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert rows[4:, 0].tolist() == [103.0, 104.0, 105.0, 106.0]
    assert torch.equal(rows[:, 0:1].expand(-1, 32), rows)


def test_window_rows_slice_noise_the_same_way_as_latents() -> None:
    _, noise = draw_refine_noise((1, 24, 2, 2, 2), (2, 32, 9), 1)
    rows = window_audio_rows(noise, 2, 5)
    assert torch.equal(rows[:5], noise[0, :, 2:7].T)
    assert torch.equal(rows[5:], noise[1, :, 2:7].T)


@pytest.mark.parametrize(
    ("index", "count", "expected"),
    [
        (0, 1, ("first", "last")),
        (0, 3, ("first",)),
        (1, 3, ()),
        (2, 3, ("last",)),
        (1, 2, ("last",)),
    ],
)
def test_a_window_holds_the_first_keyframe_only_at_the_start_and_the_last_only_at_the_end(index: int, count: int, expected: tuple[str, ...]) -> None:
    held = window_conditioning((_FIRST, _LAST), ("first", "last"), index, count)
    assert held.anchors == expected
    assert len(held.keyframes) == len(expected)
    assert all(torch.equal(frame, {"first": _FIRST, "last": _LAST}[anchor]) for frame, anchor in zip(held.keyframes, held.anchors, strict=True))


def test_a_window_without_keyframes_holds_nothing() -> None:
    held = window_conditioning((), (), 0, 1)
    assert held.keyframes == ()
    assert held.anchors == ()


def _parts(windows, value=None):
    return [torch.full((1, 24, end - start, 2, 2), float(i if value is None else value)) for i, (start, end) in enumerate(windows)]


def test_constant_parts_stitch_to_a_constant() -> None:
    windows = h3_refine_windows(72)
    out = stitch(_parts(windows, value=3.5), windows, 72, H3_REFINE_OVERLAP_LATENT_FRAMES)
    assert out.shape == (1, 24, 72, 2, 2)
    assert torch.allclose(out, torch.full_like(out, 3.5))


def test_frames_covered_by_one_window_keep_that_windows_values() -> None:
    windows = h3_refine_windows(37)  # (0, 22), (15, 37): frames 0-14 and 22-36 are covered once
    out = stitch(_parts(windows), windows, 37, H3_REFINE_OVERLAP_LATENT_FRAMES)
    assert torch.equal(out[:, :, :15], torch.zeros(1, 24, 15, 2, 2))
    assert torch.equal(out[:, :, 22:], torch.ones(1, 24, 15, 2, 2))


def test_the_overlap_fades_linearly_from_one_window_to_the_next() -> None:
    windows = h3_refine_windows(37)
    out = stitch(_parts(windows), windows, 37, H3_REFINE_OVERLAP_LATENT_FRAMES)
    ramp = torch.linspace(0, 1, 9)[1:-1]
    assert torch.allclose(out[0, 0, 15:22, 0, 0], ramp)


def test_a_single_window_is_returned_unchanged() -> None:
    part = torch.randn(1, 24, 7, 2, 2)
    out = stitch([part], [(0, 7)], 7, H3_REFINE_OVERLAP_LATENT_FRAMES)
    assert torch.equal(out, part)


def test_a_pulled_back_last_window_is_normalised_by_the_weight_sum() -> None:
    windows = h3_refine_windows(27)  # (0, 22), (5, 27): the overlap is 17 frames, more than 7
    parts = _parts(windows, value=2.0)
    out = stitch(parts, windows, 27, H3_REFINE_OVERLAP_LATENT_FRAMES)
    assert torch.allclose(out, torch.full_like(out, 2.0))
    varied = [torch.zeros(1, 24, 22, 2, 2), torch.ones(1, 24, 22, 2, 2)]
    blended = stitch(varied, windows, 27, H3_REFINE_OVERLAP_LATENT_FRAMES)
    assert torch.all((blended >= 0) & (blended <= 1))
    assert torch.equal(blended[:, :, :5], torch.zeros(1, 24, 5, 2, 2))
    assert torch.equal(blended[:, :, 22:], torch.ones(1, 24, 5, 2, 2))
    assert torch.all(torch.diff(blended[0, 0, 5:22, 0, 0]) >= 0)
