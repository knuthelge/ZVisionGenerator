"""Frame and size alignment of video jobs: the rule-driven frame count, its warning and the combined helper."""

from __future__ import annotations

import warnings

import pytest

from zvisiongenerator.core.video_family import ANY_FRAMES, FrameRule
from zvisiongenerator.utils.alignment import align_frames, align_resolution, aligned_video_dims

_LTX_RULE = FrameRule(8, 1, "nearest")
_H3_RULE = FrameRule(17, 5, "up")

# G2: every input 1..400 grouped by the count the LTX rule (alignment 8, offset 1, nearest) gives it, as aligned -> (first input, last input).
_LTX_RUNS = {
    9: (1, 12),
    17: (13, 21),
    25: (22, 28),
    33: (29, 37),
    41: (38, 44),
    49: (45, 53),
    57: (54, 60),
    65: (61, 69),
    73: (70, 76),
    81: (77, 85),
    89: (86, 92),
    97: (93, 101),
    105: (102, 108),
    113: (109, 117),
    121: (118, 124),
    129: (125, 133),
    137: (134, 140),
    145: (141, 149),
    153: (150, 156),
    161: (157, 165),
    169: (166, 172),
    177: (173, 181),
    185: (182, 188),
    193: (189, 197),
    201: (198, 204),
    209: (205, 213),
    217: (214, 220),
    225: (221, 229),
    233: (230, 236),
    241: (237, 245),
    249: (246, 252),
    257: (253, 261),
    265: (262, 268),
    273: (269, 277),
    281: (278, 284),
    289: (285, 293),
    297: (294, 300),
    305: (301, 309),
    313: (310, 316),
    321: (317, 325),
    329: (326, 332),
    337: (333, 341),
    345: (342, 348),
    353: (349, 357),
    361: (358, 364),
    369: (365, 373),
    377: (374, 380),
    385: (381, 389),
    393: (390, 396),
    401: (397, 400),
}


def _align_ltx(frames: int, alignment: int = 8) -> int:
    return align_frames(frames, FrameRule(alignment, 1, "nearest"))


def test_ltx_rule_maps_every_input_to_the_pinned_count():
    covered = [frames for first, last in _LTX_RUNS.values() for frames in range(first, last + 1)]
    assert sorted(covered) == list(range(1, 401))
    for aligned, (first, last) in _LTX_RUNS.items():
        assert {_align_ltx(frames) for frames in range(first, last + 1)} == {aligned}


def test_alignment_one_only_lifts_a_single_frame_to_two():
    assert {frames: _align_ltx(frames, 1) for frames in range(1, 401)} == {1: 2, **{frames: frames for frames in range(2, 401)}}


def test_any_frames_only_lifts_a_single_frame_to_two():
    assert {frames: align_frames(frames, ANY_FRAMES) for frames in range(1, 401)} == {1: 2, **{frames: frames for frames in range(2, 401)}}


@pytest.mark.parametrize(
    ("frames", "expected"),
    [(1, 22), (21, 22), (22, 22), (23, 39), (121, 124), (124, 124), (125, 141), (240, 243), (400, 413)],
)
def test_the_h3_rule_rounds_up_to_17k_plus_5_without_a_cap(frames, expected):
    assert align_frames(frames, _H3_RULE) == expected


def test_the_nearest_rule_rounds_halves_to_even():
    # (13 - 1) / 8 = 1.5 -> 2 and (21 - 1) / 8 = 2.5 -> 2.
    assert (align_frames(13, _LTX_RULE), align_frames(21, _LTX_RULE)) == (17, 17)


def test_a_valid_clip_is_unchanged_and_silent():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert aligned_video_dims(704, 448, 124, upscale=False, resolution_alignment=32, frame_rule=_H3_RULE, family="H3") == (704, 448, 124)

    assert seen == []


def test_a_frame_correction_warns_and_returns_the_corrected_count():
    with pytest.warns(UserWarning) as seen:
        assert aligned_video_dims(704, 448, 121, upscale=False, resolution_alignment=32, frame_rule=_H3_RULE, family="H3") == (704, 448, 124)

    assert len(seen) == 1
    assert seen[0].filename.endswith("alignment.py")  # attributed to the helper, not to the warnings module


def test_the_models_resolution_alignment_rounds_both_sides_and_the_frames():
    with pytest.warns(UserWarning):
        assert aligned_video_dims(720, 450, 52, upscale=False, resolution_alignment=32, frame_rule=_LTX_RULE, family="LTX") == (736, 448, 49)


def test_an_upscaled_clip_aligns_the_size_to_64():
    with pytest.warns(UserWarning):
        assert aligned_video_dims(720, 450, 52, upscale=True, resolution_alignment=32, frame_rule=_LTX_RULE, family="LTX") == (704, 448, 49)


def test_a_size_on_the_grid_is_unchanged_and_silent():
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert align_resolution(704, 480, 32, "Test") == (704, 480)

    assert seen == []


@pytest.mark.parametrize(("size", "expected"), [((700, 475), (704, 480)), ((710, 490), (704, 480)), ((100, 100), (96, 96))])
def test_a_size_off_the_grid_rounds_to_the_nearest_multiple(size, expected):
    with pytest.warns(UserWarning):
        assert align_resolution(*size, 32, "Test") == expected
