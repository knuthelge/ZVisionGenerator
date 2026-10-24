"""The clip a Web UI video job runs at: presets, then the model's frame and size alignment."""

from __future__ import annotations

import pytest

from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.web.video_clip import VideoClip, resolve_video_clip

_LTX_INFO = VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)

_PRESET = {"ratio": "16:9", "size": "m", "steps": None, "width": None, "height": None, "frames": None, "upscale": False}


# G7: resolve_video_clip for the preset, a custom size and an upscaled clip of an LTX model
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
def test_ltx_clip_is_unchanged(overrides, expected):
    assert resolve_video_clip(load_config(), _LTX_INFO, **{**_PRESET, **overrides}) == expected


def test_quiet_clip_resolution_reports_no_correction():
    import warnings

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        resolve_video_clip(load_config(), _LTX_INFO, **{**_PRESET, "width": 720, "height": 450, "frames": 52, "quiet": True})

    assert seen == []


def _config_with_a_17n5_family() -> dict:
    config = load_config()
    config["video_model_presets"]["h3"] = {
        "label": "MiniMax H3",
        "default_steps": 8,
        "frame_rule": {"alignment": 17, "offset": 5, "round": "up"},
        "extend_frame_step": 17,
        "carry_frames": 22,
        "capabilities": {"i2v": True, "end_frame": True, "extend": True, "upscale": True, "upscale_review": True, "lora": True, "audio": True, "low_memory": False},
    }
    return config


@pytest.mark.filterwarnings("ignore:.*Adjusted")
def test_a_clip_is_aligned_by_its_familys_frame_rule():
    info = VideoModelInfo(family="h3", backend="h3", supports_i2v=True, default_fps=24, frame_alignment=17, resolution_alignment=32)

    clip = resolve_video_clip(_config_with_a_17n5_family(), info, **{**_PRESET, "frames": 121})

    assert clip.num_frames == 124


@pytest.mark.filterwarnings("ignore:.*Adjusted")
def test_a_model_without_a_family_description_keeps_any_frame_count():
    info = VideoModelInfo(family="wan", backend="wan", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)

    assert resolve_video_clip(load_config(), info, **{**_PRESET, "frames": 52}).num_frames == 52
