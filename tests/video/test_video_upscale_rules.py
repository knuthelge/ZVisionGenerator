"""Tests for the LTX video file upscale rules: which clips it takes, its steps, its menu option and how its progress is counted."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.memory_verdict import MemoryJob, MemoryWeights, judge
from zvisiongenerator.utils.model_memory import GIB, LtxMlxWeights, MemoryBudget
from zvisiongenerator.utils.upscale import SEEDVR2_MEMORY_WARNING, WARNING_MEMORY
from zvisiongenerator.utils.video_upscale import (
    DEFAULT_LTX_UPSCALE_STEPS,
    ltx_upscale_eta,
    ltx_upscale_problem,
    ltx_upscale_prompt,
    ltx_upscale_steps,
    ltx_upscale_verdict,
    ltx_video_option,
    upscale_progress_units,
)
from zvisiongenerator.utils.weight_totals import ResolvedWeights


@pytest.mark.parametrize(("width", "height", "frames"), [(768, 448, 121), (1024, 576, 249), (32, 32, 9)])
def test_a_clip_on_the_ltx_grid_can_be_upscaled(width, height, frames):
    assert ltx_upscale_problem(width, height, frames) is None


@pytest.mark.parametrize(("width", "height"), [(770, 448), (768, 450)])
def test_a_side_off_the_32_pixel_grid_is_refused(width, height):
    assert "multiple of 32" in ltx_upscale_problem(width, height, 121)


@pytest.mark.parametrize("frames", [1, 8, 120, 124])
def test_a_frame_count_that_is_not_8k_plus_1_is_refused(frames):
    assert "8k+1" in ltx_upscale_problem(768, 448, frames)


@pytest.mark.parametrize(("steps", "units"), [(1, 3), (2, 4), (3, 5)])
def test_progress_counts_the_encode_each_step_and_the_decode(steps, units):
    assert upscale_progress_units(steps) == units


def _config(steps) -> dict:
    return {"video_model_presets": {"ltx": {"upscale": {"existing_video_steps": steps}}}}


@pytest.mark.parametrize(
    ("scene", "prompt", "extended", "expected"),
    [
        ("A harbour.", "A man walks.", False, "A harbour. A man walks."),
        (None, "A man walks.", False, "A man walks."),
        ("A harbour.", "He rides away.", True, "A harbour."),
        (None, "He rides away.", True, ""),
        (None, None, False, ""),
    ],
)
def test_the_refine_prompt_is_the_scene_then_the_prompt_and_only_the_scene_for_an_extended_clip(scene, prompt, extended, expected):
    assert ltx_upscale_prompt(scene, prompt, extended=extended) == expected


def test_the_shipped_config_refines_in_two_steps():
    assert ltx_upscale_steps(load_config()) == 2


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_steps_from_one_to_three_are_taken_from_the_config(steps):
    assert ltx_upscale_steps(_config(steps)) == steps


@pytest.mark.parametrize("config", [{}, {"video_model_presets": {"ltx": {}}}, {"video_model_presets": {"ltx": {"upscale": {"default_upscale_steps": 8}}}}])
def test_a_config_without_the_key_uses_the_default(config):
    assert ltx_upscale_steps(config) == DEFAULT_LTX_UPSCALE_STEPS


@pytest.mark.parametrize("steps", [0, 4, 2.0, True, "2"])
def test_steps_outside_one_to_three_are_refused(steps):
    with pytest.raises(ValueError, match="existing_video_steps"):
        ltx_upscale_steps(_config(steps))


def _option(monkeypatch, status, *, frame_count=121, budget="budget"):
    calls = []
    monkeypatch.setattr(
        "zvisiongenerator.utils.video_upscale.ltx_upscale_verdict", lambda *args, **kwargs: calls.append((args, kwargs)) or (None if status is None else SimpleNamespace(status=status))
    )
    option = ltx_video_option(768, 448, frame_count, model_ref="o/ltx@a", model_name="ltx-4", budget=budget)
    return option, calls


def test_the_option_is_twice_the_size_and_checks_a_two_stage_render_at_that_size(monkeypatch):
    option, calls = _option(monkeypatch, "fits")

    assert (option.factor, option.width, option.height, option.allowed, option.warning) == (2, 1536, 896, True, None)
    ((args, kwargs),) = calls
    assert args == ("o/ltx@a", "ltx-4", 1536, 896, 121)
    assert kwargs["budget"] == "budget"


def test_a_clip_whose_weights_cannot_be_sized_has_no_warning(monkeypatch):
    option, _ = _option(monkeypatch, None)

    assert (option.allowed, option.warning) == (True, None)


@pytest.mark.parametrize("status", ["tight", "too_large"])
def test_the_option_is_warned_but_allowed_when_the_clip_may_not_fit(monkeypatch, status):
    option, _ = _option(monkeypatch, status)

    assert (option.allowed, option.warning, option.warning_code) == (True, SEEDVR2_MEMORY_WARNING, WARNING_MEMORY)


def test_without_a_frame_count_there_is_no_check(monkeypatch):
    option, calls = _option(monkeypatch, "too_large", frame_count=None)

    assert (option.warning, calls) == (None, [])


class TestLtxUpscaleVerdict:
    _BUDGET = MemoryBudget(gpu_bytes=20 * GIB, unified_bytes=32 * GIB)
    _PACK = LtxMlxWeights(gemma=5 * GIB, connector=GIB, transformer=10 * GIB, decoders=GIB, upsampler=GIB)

    def _verdict(self, monkeypatch, budget=_BUDGET, resolved=ResolvedWeights(MemoryWeights(ltx_mlx=_PACK), family="ltx")):
        monkeypatch.setattr("zvisiongenerator.utils.weight_totals.resolve_ltx_weights", lambda *_args, **_kwargs: resolved)
        return ltx_upscale_verdict("o/ltx@a", "ltx-4", 1536, 896, 121, budget=budget)

    def test_it_is_the_engines_verdict_of_a_two_stage_render_at_the_output_size(self, monkeypatch):
        verdict = self._verdict(monkeypatch)

        expected = judge(MemoryJob(media="video", family="ltx", width=1536, height=896, frames=121, low_memory=True, upscale=True), self._BUDGET, MemoryWeights(ltx_mlx=self._PACK))
        assert verdict == expected

    def test_a_smaller_machine_judges_the_same_clip_less_kindly(self, monkeypatch):
        roomy = self._verdict(monkeypatch)
        small = self._verdict(monkeypatch, budget=MemoryBudget(gpu_bytes=8 * GIB, unified_bytes=12 * GIB))

        assert (roomy.status, small.status) == ("fits", "too_large")

    def test_there_is_no_verdict_without_a_budget_or_weights(self, monkeypatch):
        assert self._verdict(monkeypatch, budget=None) is None
        assert self._verdict(monkeypatch, resolved=None) is None
        assert self._verdict(monkeypatch, resolved=ResolvedWeights(MemoryWeights(), family="ltx")) is None


@pytest.mark.parametrize("finished", [[], [18.0]])
def test_there_is_no_estimate_before_a_refine_step_has_finished(finished):
    assert ltx_upscale_eta(finished, 4) is None


def test_the_estimate_counts_each_step_left_and_half_a_step_for_the_decode():
    # Encode at 18 s, two of three steps at 168 s and 318 s: 150 s a step, one step and the decode left.
    assert ltx_upscale_eta([18.0, 168.0, 318.0], 5) == pytest.approx(150 * 1.5)


def test_after_the_last_step_only_the_decode_is_left():
    assert ltx_upscale_eta([18.0, 168.0, 318.0], 4) == pytest.approx(75.0)


def test_nothing_is_left_once_every_unit_has_finished():
    assert ltx_upscale_eta([18.0, 168.0, 318.0, 400.0], 4) == 0.0
