"""The video family model: its description in the config, step settling, and the LTX behaviour it must keep."""

from __future__ import annotations

import argparse
import re
import warnings
from types import SimpleNamespace

import pytest

from _video_helpers import make_ltx_info

from zvisiongenerator.core.video_family import ANY_FRAMES, FrameRule, VideoCapabilities, VideoFamily, settle_video_steps
from zvisiongenerator.utils.config import load_config, video_family
from zvisiongenerator.utils.video_extend import resolve_extend_frames
from zvisiongenerator.utils.video_model_detect import detect_video_model
from zvisiongenerator.video_cli import _resolve_upscale_steps
from zvisiongenerator.web import server as web_server

_LTX_INFO = make_ltx_info()
_OTHER_INFO = make_ltx_info(family="wan", backend="wan", frame_alignment=1)


# ---------------------------------------------------------------------------
# G3: step behaviour of the CLI and the server for LTX
# ---------------------------------------------------------------------------

_STEP_TABLE = [
    # (upscale, steps, steps given explicitly, family, settled steps, settled upscale steps, capped from)
    (None, 8, True, "ltx", 8, None, None),
    (None, 4, True, "ltx", 4, None, None),
    (None, 20, True, "ltx", 8, None, 20),
    (None, 20, True, "wan", 20, None, None),
    (2, 50, False, "ltx", 8, 8, None),
    (2, 5, True, "ltx", 5, 5, None),
    (2, 30, True, "ltx", 8, 8, 30),
    (2, 20, True, "wan", 20, 20, None),
]


@pytest.mark.parametrize(("upscale", "steps", "explicit", "family", "settled", "upscale_steps", "capped_from"), _STEP_TABLE)
def test_cli_step_settling_matches_the_golden_table(upscale, steps, explicit, family, settled, upscale_steps, capped_from):
    args = argparse.Namespace(upscale=upscale, steps=steps)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        _resolve_upscale_steps(args, load_config(), family, steps_explicitly_set=explicit)

    assert (args.steps, args.upscale_steps) == (settled, upscale_steps)
    assert len(seen) == (0 if capped_from is None else 1)


@pytest.mark.parametrize(("upscale", "steps", "explicit", "family", "settled", "upscale_steps", "capped_from"), _STEP_TABLE)
def test_server_step_settling_matches_the_golden_table(upscale, steps, explicit, family, settled, upscale_steps, capped_from):
    args = argparse.Namespace(upscale=upscale, steps=steps if explicit else None)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        web_server._normalize_video_steps(args, load_config(), _LTX_INFO if family == "ltx" else _OTHER_INFO)

    assert (args.steps, args.upscale_steps) == (settled, upscale_steps)
    assert seen == []


def test_server_refuses_fewer_than_one_step():
    args = argparse.Namespace(upscale=None, steps=0)

    with pytest.raises(ValueError, match=r"^Steps must be at least 1\.$"):
        web_server._normalize_video_steps(args, load_config(), _LTX_INFO)


def _config_with_h3_upscale(steps: int) -> dict:
    config = load_config()
    config["video_model_presets"]["h3"]["upscale"] = {"default_upscale_steps": steps}
    config["video_model_presets"]["ltx"]["upscale"] = {"default_upscale_steps": 6}
    return config


@pytest.mark.parametrize(("family", "expected"), [("h3", 4), ("ltx", 6)])
def test_the_default_upscale_steps_come_from_the_models_own_family(family, expected):
    config = _config_with_h3_upscale(4)
    info = make_ltx_info(family=family, backend=family)
    cli_args = argparse.Namespace(upscale=2, steps=None)
    server_args = argparse.Namespace(upscale=2, steps=None)

    _resolve_upscale_steps(cli_args, config, family, steps_explicitly_set=False)
    web_server._normalize_video_steps(server_args, config, info)

    # H3 offers 4 and 8 steps, LTX is capped at 8: the family's default is settled by its own rules.
    assert (cli_args.steps, cli_args.upscale_steps) == (server_args.steps, server_args.upscale_steps) == (expected, expected)


def test_a_family_without_an_upscale_block_upscales_at_its_default_steps():
    assert video_family(load_config(), "h3").default_upscale_steps == 8
    assert video_family(load_config(), "ltx").default_upscale_steps == 8


# ---------------------------------------------------------------------------
# G7: what the LTX Web UI defaults and the extend length arithmetic give
# ---------------------------------------------------------------------------


def _bootstrap_config() -> SimpleNamespace:
    config = load_config()
    return SimpleNamespace(app_config=config, video_ratios=tuple(config["video_sizes"]), video_size_options={ratio: tuple(sizes) for ratio, sizes in config["video_sizes"].items()})


@pytest.mark.parametrize(
    ("given", "strict", "expected"),
    [
        ({"frames": None, "seconds": None, "fps": 24.0}, False, (48, None)),
        ({"frames": None, "seconds": None, "fps": 24.0}, True, (48, None)),
        ({"frames": 58, "seconds": None, "fps": 24.0}, False, (56, "--extend-frames 58 rounded to 56 (a multiple of 8).")),
        ({"frames": 64, "seconds": None, "fps": 24.0}, True, (64, None)),
        ({"frames": 4, "seconds": None, "fps": 24.0}, False, (8, "--extend-frames 4 rounded to 8 (a multiple of 8).")),
        ({"frames": None, "seconds": 2.0, "fps": 24.0}, False, (48, None)),
        ({"frames": None, "seconds": 2.5, "fps": 30.0}, False, (72, "--extend-seconds 2.5 is 75 frames at 30 fps; using 72 frames (2.40 s).")),
        ({"frames": None, "seconds": 2.5, "fps": 30.0}, True, (72, None)),
        ({"frames": 12, "seconds": 9, "fps": 24.0}, False, (16, "--extend-frames 12 rounded to 16 (a multiple of 8).")),
    ],
)
def test_ltx_extend_length_table_is_unchanged(given, strict, expected):
    assert resolve_extend_frames(strict=strict, frames_name="--extend-frames", seconds_name="--extend-seconds", **given) == expected


@pytest.mark.parametrize("frames", [58, 4, 12])
def test_strict_extend_length_refuses_a_non_multiple_of_eight(frames):
    with pytest.raises(ValueError, match=r"^--extend-frames must be a multiple of 8, at least 8\.$"):
        resolve_extend_frames(frames=frames, seconds=None, fps=24.0, strict=True, frames_name="--extend-frames")


# ---------------------------------------------------------------------------
# R2: the family description in the config
# ---------------------------------------------------------------------------


def _h3_block() -> dict:
    return {
        "label": "MiniMax H3",
        "default_steps": 8,
        "max_steps": 8,
        "step_options": [4, 8],
        "default_frames": 124,
        "frame_rule": {"alignment": 17, "offset": 5, "round": "up"},
        "extend_frame_step": 17,
        "carry_frames": 22,
        "capabilities": {"i2v": True, "end_frame": True, "extend": True, "upscale": True, "upscale_review": True, "lora": True, "audio": True, "low_memory": False},
        "platform_capabilities": {"darwin": {"upscale": False, "upscale_review": False}},
    }


def _config_with(block: dict) -> dict:
    return {"video_model_presets": {"h3": block}}


def test_the_packaged_ltx_family_is_described_literally():
    assert video_family(load_config(), "ltx") == VideoFamily(
        name="ltx",
        label="LTX distilled model",
        frame_rule=FrameRule(8, 1, "nearest"),
        capabilities=VideoCapabilities(i2v=True, end_frame=True, extend=True, upscale=True, upscale_review=True, lora=True, audio=True, low_memory=True),
        platform_capabilities={},
        default_steps=8,
        max_steps=8,
        step_options=None,
        default_frames=None,
        extend_frame_step=8,
        carry_frames=25,
        default_upscale_steps=8,
        extend_fps=None,
        platform_carry_frames={},
    )


def test_a_family_without_a_preset_block_is_none():
    assert video_family(load_config(), "wan") is None
    assert video_family({}, "ltx") is None


def test_a_family_reads_its_block_without_a_versions_overlay():
    config = load_config()
    config["video_model_presets"]["ltx"]["versions"] = {"ltx25": {"aliases": ["ltx25-4"], "label": "Other"}}

    assert video_family(config, "ltx").label == "LTX distilled model"


def test_a_full_family_block_parses():
    family = video_family(_config_with(_h3_block()), "h3")

    assert family == VideoFamily(
        name="h3",
        label="MiniMax H3",
        frame_rule=FrameRule(17, 5, "up"),
        capabilities=VideoCapabilities(i2v=True, end_frame=True, extend=True, upscale=True, upscale_review=True, lora=True, audio=True, low_memory=False),
        platform_capabilities={"darwin": {"upscale": False, "upscale_review": False}},
        default_steps=8,
        max_steps=8,
        step_options=(4, 8),
        default_frames=124,
        extend_frame_step=17,
        carry_frames=22,
        default_upscale_steps=8,
        extend_fps=None,
    )


def test_extend_fps_is_read_from_the_block_and_the_packaged_families_declare_it():
    assert video_family(_config_with({**_h3_block(), "extend_fps": 24}), "h3").extend_fps == 24
    assert video_family(load_config(), "h3").extend_fps == 24
    assert video_family(load_config(), "ltx").extend_fps is None


def _without(key: str):
    block = _h3_block()
    del block[key]
    return block


def _with(path: str, value):
    block = _h3_block()
    node = block
    *parents, last = path.split(".")
    for parent in parents:
        node = node[parent]
    node[last] = value
    return block


@pytest.mark.parametrize(
    ("block", "path"),
    [
        (_without("label"), "label"),  # a missing key
        (_without("capabilities"), "capabilities"),
        (_with("default_steps", 0), "default_steps"),  # a value out of range
        (_with("max_steps", "8"), "max_steps"),  # a wrong type
        (_with("extend_frame_step", True), "extend_frame_step"),  # a bool posing as an int
        (_with("upscale", {"default_upscale_steps": 0}), "upscale.default_upscale_steps"),  # a nested path
        (_with("frame_rule.alignment", 0), "frame_rule.alignment"),
        (_with("frame_rule.extra", 1), "frame_rule.extra"),  # an unknown key
        (_with("capabilities", {"i2v": True}), "capabilities.end_frame"),
        (_with("platform_capabilities", {"darwin": {"upscale": 0}}), "platform_capabilities.darwin.upscale"),
        (_with("step_options", [8, 4]), "step_options"),  # out of order
        (_with("default_frames", 121), "default_frames"),  # off the frame rule
    ],
)
def test_a_malformed_family_block_names_the_key_path(block, path):
    with pytest.raises(ValueError, match=rf"^{re.escape('video_model_presets.h3.' + path)}(?=\s)"):
        video_family(_config_with(block), "h3")


def test_detection_alignment_equals_the_rule_alignment():
    family = video_family(load_config(), "ltx")

    for model in ("dgrauet/ltx-2.3-mlx-q4", "unsloth/ltx-2.5-diffusers", "marcelofso/ltx-2.5-mlx-q4-fast"):
        assert detect_video_model(model).frame_alignment == family.frame_rule.alignment


# ---------------------------------------------------------------------------
# R4: settle_video_steps
# ---------------------------------------------------------------------------


def _family(**overrides) -> VideoFamily:
    return video_family(_config_with({**_h3_block(), **overrides}), "h3")


@pytest.mark.parametrize(
    ("steps", "settled", "noted"),
    [(None, 8, False), (8, 8, False), (4, 4, False), (5, 4, True), (6, 8, True), (12, 8, True), (1, 4, True)],
)
def test_h3_steps_snap_to_the_nearest_option_and_ties_go_up(steps, settled, noted):
    result = settle_video_steps(steps, upscale=False, family=_family(), default_upscale_steps=8)

    assert (result.steps, result.upscale_steps, result.note is not None) == (settled, None, noted)


def test_an_upscaled_clip_runs_its_upscale_pass_at_the_settled_steps():
    result = settle_video_steps(None, upscale=True, family=_family(), default_upscale_steps=6)

    assert (result.steps, result.upscale_steps, result.note is not None) == (8, 8, True)


def test_the_cap_applies_when_there_are_no_step_options():
    settled = settle_video_steps(20, upscale=True, family=video_family(load_config(), "ltx"), default_upscale_steps=8)

    assert (settled.steps, settled.upscale_steps, settled.note is not None) == (8, 8, True)


def test_without_a_family_nothing_is_capped():
    settled = settle_video_steps(50, upscale=False, family=None, default_upscale_steps=8)

    assert (settled.steps, settled.upscale_steps, settled.note) == (50, None, None)
    assert settle_video_steps(None, upscale=False, family=None, default_upscale_steps=8).steps is None


def test_any_frames_is_the_rule_of_an_undescribed_model():
    assert ANY_FRAMES == FrameRule(1, 1, "nearest")
    assert ANY_FRAMES.to_json() == {"alignment": 1, "offset": 1, "round": "nearest"}


def test_the_carried_frames_match_what_the_backends_carry():
    from zvisiongenerator.utils.h3_memory import H3_TAIL_FRAMES
    from zvisiongenerator.utils.video_extend import CARRY_FRAMES

    config = load_config()

    assert video_family(config, "ltx").carry_frames == CARRY_FRAMES == 25
    assert video_family(config, "h3").carry_frames == H3_TAIL_FRAMES == 22


# ---------------------------------------------------------------------------
# The frames an extension carries, per platform
# ---------------------------------------------------------------------------


def test_the_packaged_h3_carries_one_frame_on_macos_and_22_elsewhere():
    from zvisiongenerator.backends import video_family_on_platform
    from zvisiongenerator.utils.video_extend import window_frames

    config = load_config()
    described = video_family(config, "h3")

    assert (described.carry_frames, described.platform_carry_frames) == (22, {"darwin": 1})
    carried = {}
    for platform in ("darwin", "linux", "win32"):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("zvisiongenerator.backends.sys.platform", platform)
            family = video_family_on_platform(config, "h3")
            carried[platform] = (family.carry_frames, window_frames(51, family), window_frames(17, family))
    assert carried == {"darwin": (1, 56, 22), "linux": (22, 73, 39), "win32": (22, 73, 39)}


def test_a_family_with_no_platform_carry_is_the_same_on_every_platform(monkeypatch):
    from zvisiongenerator.backends import video_family_on_platform

    monkeypatch.setattr("zvisiongenerator.backends.sys.platform", "darwin")

    assert video_family_on_platform(load_config(), "ltx") == video_family(load_config(), "ltx")
    assert video_family_on_platform(load_config(), "wan") is None


@pytest.mark.parametrize(
    ("value", "path"),
    [("one", "platform_carry_frames"), ({"darwin": 0}, "platform_carry_frames.darwin")],
)
def test_a_malformed_platform_carry_names_its_key(value, path):
    with pytest.raises(ValueError, match=re.escape(path)):
        video_family(_config_with({**_h3_block(), "platform_carry_frames": value}), "h3")
