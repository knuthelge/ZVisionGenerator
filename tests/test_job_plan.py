"""Tests for the job plan value types."""

from __future__ import annotations

import dataclasses

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

    def test_types_are_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _iteration().seed = 1
        with pytest.raises(dataclasses.FrozenInstanceError):
            JobPlan(iterations=()).cancelled = True

    def test_defaults(self):
        iteration = _iteration()
        assert iteration.enhanced_prompt is None and iteration.enhance_status is EnhanceStatus.OFF
        assert JobPlan(iterations=()).cancelled is False

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
    def test_requested_without_rewrite_is_valid(self, status):
        assert _iteration(enhance=EnhanceSettings(), enhance_status=status).enhance_status is status
