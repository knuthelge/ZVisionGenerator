"""Tests for the job plan value types."""

from __future__ import annotations


import pytest

from zvisiongenerator.core.job_plan import EnhanceStatus, IterationPlan, JobPlan
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings


def _iteration(**overrides) -> IterationPlan:
    values = dict(run_index=0, set_name="a", prompt_index=0, prompt="a fox", seed=7, resolved_prompt="a fox", enhance=None)
    values.update(overrides)
    return IterationPlan(**values)


class TestJobPlan:
    def test_enum_values(self):
        assert [status.value for status in EnhanceStatus] == ["off", "enhanced", "failed", "skipped"]

    def test_has_rewrites(self):
        enhanced = _iteration(enhance=EnhanceSettings(), enhanced_prompt="A fox in snow.", enhance_status=EnhanceStatus.ENHANCED)
        failed = _iteration(enhance=EnhanceSettings(), enhance_status=EnhanceStatus.FAILED)
        assert JobPlan(iterations=(_iteration(), enhanced)).has_rewrites
        assert not JobPlan(iterations=(_iteration(), failed)).has_rewrites
        assert not JobPlan(iterations=()).has_rewrites

    @pytest.mark.parametrize(
        "overrides",
        [
            {"enhance": EnhanceSettings(), "enhance_status": EnhanceStatus.ENHANCED},
            {"enhance": EnhanceSettings(), "enhanced_prompt": "text", "enhance_status": EnhanceStatus.FAILED},
            {"enhance": EnhanceSettings(), "enhanced_prompt": "text"},
            {"enhance": None, "enhance_status": EnhanceStatus.SKIPPED},
            {"enhance": None, "enhanced_prompt": "text", "enhance_status": EnhanceStatus.ENHANCED},
        ],
    )
    def test_invariant_breaches_raise(self, overrides):
        with pytest.raises(ValueError, match="IterationPlan"):
            _iteration(**overrides)

    @pytest.mark.parametrize("status", [EnhanceStatus.OFF, EnhanceStatus.FAILED, EnhanceStatus.SKIPPED])
    def test_enhanced_scene_needs_the_enhanced_status(self, status):
        with pytest.raises(ValueError, match="enhanced_scene"):
            _iteration(enhance=EnhanceSettings(), enhanced_scene="A castle.", enhance_status=status)

    def test_scene_fields_ride_along_with_a_rewrite(self):
        iteration = _iteration(
            enhance=EnhanceSettings(),
            scene="A {castle|castle}.",
            resolved_scene="A castle.",
            enhanced_prompt="A knight rides in.",
            enhanced_scene="A castle at dusk.",
            enhance_status=EnhanceStatus.ENHANCED,
        )
        assert (iteration.scene, iteration.resolved_scene, iteration.enhanced_scene) == ("A {castle|castle}.", "A castle.", "A castle at dusk.")

    def test_scene_without_a_rewrite_is_valid(self):
        assert _iteration(scene="A castle.", resolved_scene="A castle.").enhanced_scene is None

    @pytest.mark.parametrize("status", [EnhanceStatus.OFF, EnhanceStatus.FAILED, EnhanceStatus.SKIPPED])
    def test_requested_without_rewrite_is_valid(self, status):
        assert _iteration(enhance=EnhanceSettings(), enhance_status=status).enhance_status is status

    def test_require_runnable(self):
        JobPlan(iterations=(_iteration(),)).require_runnable(1)
        with pytest.raises(ValueError, match="cancelled"):
            JobPlan(iterations=(_iteration(),), cancelled=True).require_runnable(1)
        with pytest.raises(ValueError, match="1 iterations, but the batch has 2"):
            JobPlan(iterations=(_iteration(),)).require_runnable(2)

    def test_iteration_for_checks_order(self):
        plan = JobPlan(iterations=(_iteration(), _iteration(prompt_index=1)))
        assert plan.iteration_for(2, run_index=0, set_name="a", prompt_index=1) is plan.iterations[1]
        with pytest.raises(RuntimeError, match="out of order"):
            plan.iteration_for(1, run_index=0, set_name="b", prompt_index=0)
