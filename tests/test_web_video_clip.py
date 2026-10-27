"""The clip a Web UI video job runs at: presets, then the model's frame and size alignment."""

from __future__ import annotations

import pytest
from _web_helpers import packaged_config

from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.web.video_clip import VideoClip, resolve_video_clip

_LTX_INFO = VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)

_PRESET = {"ratio": "16:9", "size": "m", "steps": None, "width": None, "height": None, "frames": None, "upscale": False}


# resolve_video_clip for the preset, a custom size, an upscaled clip and a portrait ratio of an LTX model
@pytest.mark.filterwarnings("ignore:.*Adjusted")
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, VideoClip(steps=8, width=1024, height=576, num_frames=121)),
        ({"steps": 4, "width": 720, "height": 450, "frames": 52}, VideoClip(steps=4, width=736, height=448, num_frames=49)),
        ({"width": 720, "height": 450, "frames": 52, "upscale": True}, VideoClip(steps=8, width=704, height=448, num_frames=49)),
        ({"ratio": "9:16", "size": "s", "frames": 100}, VideoClip(steps=8, width=448, height=768, num_frames=97)),
    ],
    ids=["preset", "custom", "upscale", "portrait"],
)
def test_an_ltx_clip_takes_the_preset_then_the_models_alignment(overrides, expected):
    assert resolve_video_clip(packaged_config(), _LTX_INFO, **{**_PRESET, **overrides}) == expected


def test_quiet_clip_resolution_reports_no_correction():
    import warnings

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        resolve_video_clip(packaged_config(), _LTX_INFO, **{**_PRESET, "width": 720, "height": 450, "frames": 52, "quiet": True})

    assert seen == []


@pytest.mark.filterwarnings("ignore:.*Adjusted")
def test_a_clip_is_aligned_by_its_familys_frame_rule():
    info = VideoModelInfo(family="h3", backend="h3", supports_i2v=True, default_fps=24, frame_alignment=17, resolution_alignment=32)

    clip = resolve_video_clip(packaged_config(), info, **{**_PRESET, "frames": 121})

    assert clip.num_frames == 124


@pytest.mark.filterwarnings("ignore:.*Adjusted")
def test_a_model_without_a_family_description_keeps_any_frame_count():
    info = VideoModelInfo(family="wan", backend="wan", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)

    assert resolve_video_clip(packaged_config(), info, **{**_PRESET, "frames": 52}).num_frames == 52
