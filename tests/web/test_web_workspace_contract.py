"""The video model payload of the Web UI workspace: frame rule, hidden controls, steps and extend step per model."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from tests._factories import load_packaged_config

from zvisiongenerator import backends
from zvisiongenerator.core.video_family import VideoCapabilities
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.web import workspace_api as workspace_api_module
from zvisiongenerator.web.workspace_api import _build_video_bootstrap_defaults
from zvisiongenerator.web.workspace_contract import (
    AUDIO_TRACK_CONTROLS,
    AUDIO_WORKFLOW_VALUES,
    CANONICAL_WORKFLOW_VALUES,
    END_FRAME_CONTROLS,
    WORKFLOW_DEFINITIONS,
    default_workflow_for_mode,
    hidden_video_controls,
    workflow_mode,
)

_UPSCALE_CONTROLS = ["video_upscale_enabled", "video_upscale_factor", "video_upscale_review"]
_END_CONTROLS = ["end_image", "end_image_clear", "end_image_path"]
_TRACK_CONTROLS = ["audio_track", "audio_track_path", "audio_track_start"]


def _web_config(config: dict | None = None) -> SimpleNamespace:
    config = config if config is not None else load_packaged_config()
    return SimpleNamespace(app_config=config, video_ratios=tuple(config["video_sizes"]), video_size_options={ratio: tuple(sizes) for ratio, sizes in config["video_sizes"].items()})


# ---------------------------------------------------------------------------
# the media mode of a workflow
# ---------------------------------------------------------------------------


def test_image_and_video_have_their_own_default_workflow() -> None:
    assert [default_workflow_for_mode(mode) for mode in ("image", "video")] == ["txt2img", "txt2vid"]
    assert workflow_mode(AUDIO_WORKFLOW_VALUES[0]) == "audio"


# ---------------------------------------------------------------------------
# hidden_video_controls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("missing", "hidden"),
    [
        (None, []),
        ("lora", ["loras"]),
        ("upscale", _UPSCALE_CONTROLS),
        ("upscale_review", ["video_upscale_review"]),
        ("low_memory", ["low_memory"]),
        ("audio", ["audio"]),
        ("end_frame", _END_CONTROLS),
        ("audio_track", _TRACK_CONTROLS),
    ],
)
def test_hidden_controls_follow_the_missing_capability(missing, hidden):
    flags = {name: name != missing for name in ("i2v", "end_frame", "extend", "upscale", "upscale_review", "lora", "audio", "low_memory", "audio_track", "audio_track_upscale")}

    assert hidden_video_controls(VideoCapabilities(**flags)) == hidden


def test_several_missing_capabilities_hide_each_control_once():
    caps = VideoCapabilities(i2v=True, end_frame=False, extend=True, upscale=False, upscale_review=False, lora=True, audio=True, low_memory=False, audio_track=False, audio_track_upscale=False)

    assert hidden_video_controls(caps) == sorted([*_END_CONTROLS, *_TRACK_CONTROLS, "low_memory", *_UPSCALE_CONTROLS])


def test_extend_and_the_other_workflow_flags_hide_no_control():
    caps = VideoCapabilities(i2v=False, end_frame=True, extend=False, upscale=True, upscale_review=True, lora=True, audio=True, low_memory=True, audio_track=True, audio_track_upscale=False)

    assert hidden_video_controls(caps) == []


# ---------------------------------------------------------------------------
# The payload of an LTX model keeps its values and gains the family keys
# ---------------------------------------------------------------------------

_LTX_PAYLOAD = {
    "ratio": "16:9",
    "size": "s",
    "steps": 8,
    "width": 768,
    "height": 448,
    "frame_count": 121,
    "audio": True,
    "low_memory": True,
    "supports_i2v": True,
    "supports_quantize": False,
    "quantize": None,
    "max_steps": 8,
    "fps": 24,
    "upscale": {"enabled": False, "factor": 2, "steps": None},
    "enhance_max_words": 300,
    "family": "ltx",
}
_LTX_FAMILY_KEYS = {
    "frame_rule": {"alignment": 8, "offset": 1, "round": "nearest"},
    "hidden_controls": [],
    "supports_extend": True,
    "supports_audio_track_upscale": True,
    "step_options": None,
    "extend_frame_step": 8,
}


@pytest.mark.parametrize("model", ["ltx-4", "ltx-8", "ltx25-4"])
def test_an_ltx_payload_keeps_its_values_and_adds_the_family_keys(model):
    assert _build_video_bootstrap_defaults(model, _web_config()) == {**_LTX_PAYLOAD, **_LTX_FAMILY_KEYS}


# ---------------------------------------------------------------------------
# An H3 payload (from the packaged config's family block)
# ---------------------------------------------------------------------------


def _patch_h3(monkeypatch, platform: str) -> None:
    info = VideoModelInfo(family="h3", backend="h3", supports_i2v=True, default_fps=24, frame_alignment=17, resolution_alignment=32)
    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(workspace_api_module, "detect_video_model", lambda _model: info)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(backends, "_VIDEO_BACKENDS_MAP", {platform: {"ltx": object, "h3": object}})


def test_an_h3_payload_carries_the_rule_the_options_and_the_hidden_controls(monkeypatch):
    _patch_h3(monkeypatch, "linux")

    payload = _build_video_bootstrap_defaults("h3", _web_config())

    assert payload["frame_rule"] == {"alignment": 17, "offset": 5, "round": "up"}
    assert payload["hidden_controls"] == ["low_memory"]
    assert (payload["supports_extend"], payload["step_options"], payload["extend_frame_step"]) == (True, [4, 8], 17)
    assert (payload["steps"], payload["max_steps"], payload["frame_count"], payload["frame_rule"]["alignment"], payload["supports_i2v"]) == (8, 8, 124, 17, True)
    assert "frame_alignment" not in payload
    assert payload["family"] == "h3"
    # H3 follows a track but can't upscale a clip made from one.
    assert payload["supports_audio_track_upscale"] is False


def test_an_ltx_payload_can_upscale_a_clip_made_from_a_track():
    assert _build_video_bootstrap_defaults("ltx-4", _web_config())["supports_audio_track_upscale"] is True


def test_an_h3_payload_on_macos_also_hides_the_upscale_controls(monkeypatch):
    _patch_h3(monkeypatch, "darwin")

    payload = _build_video_bootstrap_defaults("h3", _web_config())

    assert payload["hidden_controls"] == ["low_memory", *_UPSCALE_CONTROLS]
    assert payload["supports_extend"] is True


def test_a_default_frame_count_off_the_rule_is_aligned(monkeypatch):
    _patch_h3(monkeypatch, "linux")
    config = load_packaged_config()
    del config["video_model_presets"]["h3"]["default_frames"]

    # The size table's 121 frames is not a count of 17k + 5; the next one is.
    assert _build_video_bootstrap_defaults("h3", _web_config(config))["frame_count"] == 124


def test_a_model_without_a_family_description_keeps_the_permissive_payload(monkeypatch):
    info = VideoModelInfo(family="wan", backend="wan", supports_i2v=True, default_fps=24, frame_alignment=4, resolution_alignment=32)
    monkeypatch.setattr(workspace_api_module, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(workspace_api_module, "detect_video_model", lambda _model: info)

    payload = _build_video_bootstrap_defaults("wan", _web_config())

    assert payload["frame_rule"] == {"alignment": 4, "offset": 1, "round": "nearest"}
    # Every capability is permissive for an undescribed family except the opt-in last frame and audio track.
    assert (payload["hidden_controls"], payload["supports_extend"], payload["step_options"], payload["extend_frame_step"]) == (sorted([*_END_CONTROLS, *_TRACK_CONTROLS]), True, None, 8)


# ---------------------------------------------------------------------------
# The last-frame controls in the workflow contract
# ---------------------------------------------------------------------------


_START_IMAGE_CONTROLS = {"reference_image", "reference_image_path", "reference_image_clear"}


def test_both_video_workflows_offer_the_last_frame_controls():
    offering = [name for name in CANONICAL_WORKFLOW_VALUES if set(END_FRAME_CONTROLS) & set(WORKFLOW_DEFINITIONS[name]["visible_controls"])]

    assert offering == ["txt2vid", "img2vid"]
    for name in offering:
        assert set(END_FRAME_CONTROLS) <= set(WORKFLOW_DEFINITIONS[name]["visible_controls"])


def test_no_workflow_clears_an_image_or_its_strength():
    clearing = {name for name in CANONICAL_WORKFLOW_VALUES if {"image_path", "end_image_path", "image_strength"} & set(WORKFLOW_DEFINITIONS[name]["clear_fields"])}

    assert clearing == set()


@pytest.mark.parametrize("workflow", ["txt2img", "img2img", "txt2vid", "img2vid"])
def test_every_image_and_video_workflow_offers_the_start_image_controls(workflow):
    assert _START_IMAGE_CONTROLS <= set(WORKFLOW_DEFINITIONS[workflow]["visible_controls"])


def test_only_image_to_image_offers_the_image_strength():
    offering = [name for name in CANONICAL_WORKFLOW_VALUES if "image_strength" in WORKFLOW_DEFINITIONS[name]["visible_controls"]]

    assert offering == ["img2img"]


@pytest.mark.parametrize("workflows", [("txt2img", "img2img"), ("txt2vid", "img2vid")])
def test_the_two_workflows_of_a_tab_clear_the_same_fields(workflows):
    first, second = (WORKFLOW_DEFINITIONS[name]["clear_fields"] for name in workflows)

    assert sorted(first) == sorted(second)


@pytest.mark.parametrize("workflow", ["txt2img", "img2img"])
def test_image_workflows_show_the_quantize_control_the_reuse_param_is_gated_on(workflow):
    """The Workspace applies a reused ``quantize`` only where this control is visible, so the real contract must list it."""
    assert "quantize" in WORKFLOW_DEFINITIONS[workflow]["visible_controls"]


@pytest.mark.parametrize("workflow", ["txt2vid", "img2vid", "extend"])
def test_every_video_workflow_offers_the_audio_track_controls(workflow):
    assert set(AUDIO_TRACK_CONTROLS) <= set(WORKFLOW_DEFINITIONS[workflow]["visible_controls"])


@pytest.mark.parametrize("workflow", ["txt2img", "img2img"])
def test_the_image_workflows_offer_no_audio_track_controls(workflow):
    assert not set(AUDIO_TRACK_CONTROLS) & set(WORKFLOW_DEFINITIONS[workflow]["visible_controls"])


# ---------------------------------------------------------------------------
# Controls a workflow shows only where its definition lists them
# ---------------------------------------------------------------------------


def _workflows_listing(control: str) -> list[str]:
    return [name for name in CANONICAL_WORKFLOW_VALUES if control in WORKFLOW_DEFINITIONS[name]["visible_controls"]]


@pytest.mark.parametrize(
    ("control", "workflows"),
    [
        ("preset", ["txt2img", "img2img", "txt2vid", "img2vid", "extend"]),
        ("negative_prompt", ["txt2img", "img2img"]),
        ("json_prompt", ["txt2img", "img2img"]),
        ("first_sigma", ["txt2img", "img2img"]),
    ],
)
def test_the_contract_lists_the_controls_that_depend_on_the_workflow(control, workflows):
    """The Workspace shows these where the workflow lists them (and the model supports them): neither Edit nor Audio does."""
    assert _workflows_listing(control) == workflows
