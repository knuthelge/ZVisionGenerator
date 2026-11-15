"""Tests for the shared MiniMax H3 Ref2VA helpers: reference preparation, the task tag and the packed layout."""

from __future__ import annotations


import pytest
import torch

from tests._images import SIDEWAYS_UPRIGHT_SIZE, noise_picture, write_cutout, write_sideways_photo
from zvisiongenerator.utils.h3_ref2va import load_h3_reference, ref2va_sequence, task_tag

_PATCH = (1, 2, 2)
_AUDIO_TAG, _VIDEO_TAG = 2, 0


def _save(tmp_path, size, name="ref.png"):
    path = tmp_path / name
    noise_picture(*size).save(path)
    return path


@pytest.mark.parametrize(
    ("size", "expected"),
    [((1344, 768), (1792, 1024)), ((1000, 1500), (1024, 1536)), ((1024, 1024), (1024, 1024)), ((400, 400), (1024, 1024)), ((200, 100), (2048, 1024)), ((1920, 1080), (1824, 1024))],
)
def test_a_reference_is_scaled_to_a_short_edge_of_1024_and_a_multiple_of_32(tmp_path, size, expected):
    image = load_h3_reference(_save(tmp_path, size))
    assert image.size == expected
    assert image.mode == "RGB"


def test_a_reference_is_turned_upright_by_its_exif_orientation(tmp_path):
    image = load_h3_reference(write_sideways_photo(tmp_path / "phone.jpg"))
    width, height = SIDEWAYS_UPRIGHT_SIZE
    assert image.size == (1024, round(height * 1024 / width / 32) * 32)
    top, bottom = image.getpixel((512, 100)), image.getpixel((512, image.height - 100))
    assert top[0] > 200 > top[2]  # red on top
    assert bottom[2] > 200 > bottom[0]  # blue below


def test_a_transparent_reference_is_flattened_onto_white(tmp_path):
    image = load_h3_reference(write_cutout(tmp_path / "cutout.png"))
    assert image.mode == "RGB"
    assert image.getpixel((1700, 500)) == (255, 255, 255)
    assert image.getpixel((300, 500))[0] > 200


def test_a_missing_reference_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_h3_reference(tmp_path / "absent.png")


@pytest.mark.parametrize(
    ("mode", "audio_track", "expected"),
    [
        ("t2v", False, "[reference generation]"),
        ("t2v", True, "[reference generation + audio reuse]"),
        ("i2v", False, "[reference generation + keyframe completion]"),
        ("i2v", True, "[reference generation + keyframe completion + audio reuse]"),
        ("extend", False, "[video continuation + reference generation] The target video continues <Video 1> as one continuous shot with no cut."),
        (
            "extend",
            True,
            "[video continuation + reference generation + audio reuse] The target video continues <Video 1> as one continuous shot with no cut.",
        ),
    ],
)
def test_task_tag_is_the_fixed_string_of_each_mode(mode, audio_track, expected):
    assert task_tag(mode, audio_track) == expected


_TEXT_TAGS = [1, 1, 0, 0, 0, 1, 1, 1]
_SHAPES = [(1, 4, 6), (1, 6, 4)]
_GEOMETRY = {"latent_frames": 2, "latent_height": 4, "latent_width": 6, "audio_latents": 3}
_FRAME_ROWS = (4 // 2) * (6 // 2)
_REFERENCE_ROWS = 2 * 3 + 3 * 2


def _sequence(anchors, shapes=_SHAPES):
    return ref2va_sequence(_TEXT_TAGS, shapes, **_GEOMETRY, anchors=anchors, patch=_PATCH, audio_channels=2, audio_tag=_AUDIO_TAG, video_tag=_VIDEO_TAG)


def _diffusers_sequence(shapes=_SHAPES):
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep

    return MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        torch.tensor(_TEXT_TAGS),
        [MiniMaxH3ImageReference(image=None) for _ in shapes],
        [torch.empty(1, 1, *shape) for shape in shapes],
        [],
        2,
        4,
        6,
        3,
        _PATCH,
        2,
        _AUDIO_TAG,
        _VIDEO_TAG,
    )


def test_the_sequence_without_a_keyframe_is_diffusers_own_layout():
    sequence = _sequence(anchors=())
    position_ids, token_tags, video_indices, audio_indices, text_indices, reference_rows, _ = _diffusers_sequence()
    assert torch.equal(sequence.position_ids, position_ids)
    assert torch.equal(sequence.token_tags, token_tags)
    assert torch.equal(sequence.video_indices, video_indices)
    assert torch.equal(sequence.audio_indices, audio_indices)
    assert torch.equal(sequence.text_indices, text_indices)
    assert sequence.keyframe_rows == reference_rows == _REFERENCE_ROWS


def test_the_sequence_row_counts():
    sequence = _sequence(anchors=())
    text, audio, video = len(_TEXT_TAGS), 3 * 2, 2 * _FRAME_ROWS
    assert sequence.position_ids.shape == (text + _REFERENCE_ROWS + audio + video, 3)
    assert sequence.text_indices.tolist() == list(range(text))
    assert sequence.video_indices.numel() == _REFERENCE_ROWS + video
    assert sequence.audio_indices.numel() == audio


def test_reference_rows_come_first_in_the_video_indices():
    sequence = _sequence(anchors=())
    assert sequence.video_indices[:_REFERENCE_ROWS].tolist() == list(range(len(_TEXT_TAGS), len(_TEXT_TAGS) + _REFERENCE_ROWS))


def test_the_keyframe_sits_after_the_references_at_the_first_target_frame_positions():
    plain, spliced = _sequence(anchors=()), _sequence(anchors=("first",))
    text = len(_TEXT_TAGS)
    at = text + _REFERENCE_ROWS
    assert spliced.keyframe_rows == _REFERENCE_ROWS + _FRAME_ROWS
    assert spliced.position_ids.shape[0] == plain.position_ids.shape[0] + _FRAME_ROWS
    first_target = int(plain.video_indices[_REFERENCE_ROWS])
    assert torch.equal(spliced.position_ids[at : at + _FRAME_ROWS], plain.position_ids[first_target : first_target + _FRAME_ROWS])
    assert spliced.video_indices[_REFERENCE_ROWS : spliced.keyframe_rows].tolist() == list(range(at, at + _FRAME_ROWS))
    assert spliced.token_tags[at : at + _FRAME_ROWS].tolist() == [_VIDEO_TAG] * _FRAME_ROWS
    # The references and the text keep their rows; the audio and target video move down by the keyframe rows.
    assert torch.equal(spliced.position_ids[:at], plain.position_ids[:at])
    assert torch.equal(spliced.video_indices[:_REFERENCE_ROWS], plain.video_indices[:_REFERENCE_ROWS])
    assert torch.equal(spliced.video_indices[spliced.keyframe_rows :], plain.video_indices[_REFERENCE_ROWS:] + _FRAME_ROWS)
    assert torch.equal(spliced.text_indices, plain.text_indices)


def test_the_keyframe_shifts_the_audio_indices_and_moves_the_audio_rows():
    plain, spliced = _sequence(anchors=()), _sequence(anchors=("first",))
    assert torch.equal(spliced.audio_indices, plain.audio_indices + _FRAME_ROWS)
    assert torch.equal(spliced.position_ids[spliced.audio_indices], plain.position_ids[plain.audio_indices])
    assert torch.equal(spliced.position_ids[spliced.video_indices[spliced.keyframe_rows :]], plain.position_ids[plain.video_indices[_REFERENCE_ROWS:]])
    assert set(spliced.token_tags[spliced.audio_indices].tolist()) == {_AUDIO_TAG}


def test_a_keyframe_without_references_comes_right_after_the_text():
    sequence = _sequence(anchors=("first",), shapes=[])
    assert sequence.keyframe_rows == _FRAME_ROWS
    assert sequence.video_indices[:_FRAME_ROWS].tolist() == list(range(len(_TEXT_TAGS), len(_TEXT_TAGS) + _FRAME_ROWS))


def _last_frame_positions():
    """Return the positions fl2va gives a ``"last"`` keyframe, shifted to the rotary origin of the Ref2VA layout's first target frame."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

    text = len(_TEXT_TAGS)
    plain = MiniMaxH3PrepareLayoutStep.build_packed_sequence(torch.tensor(_TEXT_TAGS), 2, 4, 6, 3, _PATCH, 2, _AUDIO_TAG, _VIDEO_TAG, ("last",))[0]
    ref2va = _diffusers_sequence()
    positions = plain[text : text + _FRAME_ROWS].clone()
    positions[:, 0] += ref2va[0][int(ref2va[2][_REFERENCE_ROWS]), 0] - text
    return positions


def test_a_last_keyframe_takes_the_positions_of_fl2vas_last_anchor():
    sequence = _sequence(anchors=("last",))
    at = len(_TEXT_TAGS) + _REFERENCE_ROWS
    assert torch.equal(sequence.position_ids[at : at + _FRAME_ROWS], _last_frame_positions())


def test_the_first_and_the_last_keyframe_follow_the_references_in_that_order():
    plain, both = _sequence(anchors=()), _sequence(anchors=("first", "last"))
    text = len(_TEXT_TAGS)
    at = text + _REFERENCE_ROWS
    keyframe_rows = 2 * _FRAME_ROWS
    first_target = int(plain.video_indices[_REFERENCE_ROWS])
    assert both.keyframe_rows == _REFERENCE_ROWS + keyframe_rows
    assert torch.equal(both.position_ids[:at], plain.position_ids[:at])
    assert torch.equal(both.position_ids[at : at + _FRAME_ROWS], plain.position_ids[first_target : first_target + _FRAME_ROWS])
    assert torch.equal(both.position_ids[at + _FRAME_ROWS : at + keyframe_rows], _last_frame_positions())
    assert both.token_tags[at : at + keyframe_rows].tolist() == [_VIDEO_TAG] * keyframe_rows
    assert both.video_indices[_REFERENCE_ROWS : both.keyframe_rows].tolist() == list(range(at, at + keyframe_rows))
    assert torch.equal(both.video_indices[:_REFERENCE_ROWS], plain.video_indices[:_REFERENCE_ROWS])
    assert torch.equal(both.video_indices[both.keyframe_rows :], plain.video_indices[_REFERENCE_ROWS:] + keyframe_rows)
    assert torch.equal(both.audio_indices, plain.audio_indices + keyframe_rows)
    assert torch.equal(both.position_ids[both.audio_indices], plain.position_ids[plain.audio_indices])
    assert torch.equal(both.position_ids[both.video_indices[both.keyframe_rows :]], plain.position_ids[plain.video_indices[_REFERENCE_ROWS:]])
    assert torch.equal(both.text_indices, plain.text_indices)
