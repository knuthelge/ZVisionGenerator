"""Tests for the preflight phase: iteration planning, the enhancer lifecycle, controls and events."""

from __future__ import annotations

from unittest.mock import patch

from zvisiongenerator.core.job_plan import EnhanceStatus
from zvisiongenerator.preflight import plan_iterations
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings


def _plan(prompts_data, **overrides):
    values = dict(runs=1, seed=42, seed_min=4, seed_max=2**32 - 1, json_prompt=False, disabled=False, override=None, enhance_by_set=None)
    values.update(overrides)
    return plan_iterations(prompts_data, **values)


class TestPlanIterations:
    def test_order_is_runs_sets_prompts(self):
        plan = _plan({"a": [("a1", None), ("a2", None)], "b": [("b1", "neg")]}, runs=2)
        assert [(it.run_index, it.set_name, it.prompt_index, it.prompt) for it in plan] == [
            (0, "a", 0, "a1"),
            (0, "a", 1, "a2"),
            (0, "b", 0, "b1"),
            (1, "a", 0, "a1"),
            (1, "a", 1, "a2"),
            (1, "b", 0, "b1"),
        ]

    def test_fixed_seed(self):
        assert {it.seed for it in _plan({"a": [("x", None), ("y", None)]}, seed=9)} == {9}

    def test_random_seed_uses_bounds(self):
        with patch("zvisiongenerator.preflight.random.randint", side_effect=[11, 12]) as randint:
            plan = _plan({"a": [("x", None), ("y", None)]}, seed=None, seed_min=5, seed_max=50)
        assert [it.seed for it in plan] == [11, 12]
        randint.assert_called_with(5, 50)

    def test_random_choices_expanded(self):
        (iteration,) = _plan({"a": [("{red|red} fox", None)]})
        assert iteration.prompt == "{red|red} fox" and iteration.resolved_prompt == "red fox"

    def test_json_caption_verbatim_and_never_enhanced(self):
        caption = '{"a": "{x|y}"}'
        (iteration,) = _plan({"prompt": [(caption, None)]}, json_prompt=True, override=EnhanceSettings())
        assert iteration.resolved_prompt == caption and iteration.enhance is None

    def test_enhance_precedence(self):
        entry, override = EnhanceSettings(style="anime"), EnhanceSettings(style="photo")
        by_set = {"a": [entry, None]}
        data = {"a": [("x", None), ("y", None)]}
        assert [it.enhance for it in _plan(data, enhance_by_set=by_set)] == [entry, None]
        assert [it.enhance for it in _plan(data, enhance_by_set=by_set, override=override)] == [override, override]
        assert [it.enhance for it in _plan(data, enhance_by_set=by_set, override=override, disabled=True)] == [None, None]

    def test_starts_unenhanced(self):
        (iteration,) = _plan({"a": [("x", None)]}, override=EnhanceSettings())
        assert iteration.enhance_status is EnhanceStatus.OFF and iteration.enhanced_prompt is None

    def test_empty_input(self):
        assert _plan({}) == ()
        assert _plan({"a": []}) == ()
        assert _plan({"a": [("x", None)]}, runs=0) == ()
