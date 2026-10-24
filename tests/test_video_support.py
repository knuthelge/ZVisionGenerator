"""Refusals of a video request that the model's family cannot do on this platform."""

from __future__ import annotations

import pytest

import zvisiongenerator.backends as backends_module
from zvisiongenerator.utils.video_support import unsupported_video_request

_ALL_ON = {"i2v": True, "end_frame": False, "extend": True, "upscale": True, "upscale_review": True, "lora": True, "audio": True, "low_memory": True}


def _config(**capabilities: bool) -> dict:
    return {
        "video_model_presets": {
            "ltx": {
                "label": "LTX distilled model",
                "default_steps": 8,
                "frame_rule": {"alignment": 8, "offset": 1, "round": "nearest"},
                "extend_frame_step": 8,
                "capabilities": {**_ALL_ON, **capabilities},
            }
        }
    }


@pytest.fixture(autouse=True)
def _linux(monkeypatch):
    monkeypatch.setattr(backends_module.sys, "platform", "linux")


def test_a_request_the_family_can_do_is_not_refused():
    assert unsupported_video_request(_config(), "ltx", image=True, extend=True, review=True) is None


def test_a_family_the_config_does_not_describe_is_not_refused():
    assert unsupported_video_request({}, "ltx", image=True, extend=True, review=True) is None


@pytest.mark.parametrize(
    ("capability", "request_flags", "message"),
    [
        ("i2v", {"image": True}, "LTX distilled model does not support image-to-video. Pick another video model."),
        ("extend", {"extend": True}, "LTX distilled model can't extend a clip. Pick another video model."),
        ("upscale_review", {"review": True}, "LTX distilled model can't pause for review before upscaling on this platform."),
    ],
)
def test_a_missing_capability_refuses_only_the_workflow_that_needs_it(capability, request_flags, message):
    config = _config(**{capability: False})
    flags = {"image": False, "extend": False, "review": False}

    assert unsupported_video_request(config, "ltx", **{**flags, **request_flags}) == message
    assert unsupported_video_request(config, "ltx", **flags) is None


def test_a_platform_without_extend_refuses_an_extension(monkeypatch):
    monkeypatch.setattr(backends_module, "supports_video_extend", lambda: False)

    assert unsupported_video_request(_config(), "ltx", image=False, extend=True, review=False) == "LTX distilled model can't extend a clip. Pick another video model."
