"""Tests for negative prompt handling in batch generation.

Ensures effective_negative is computed before use, and that negative prompts
are correctly preserved or suppressed based on supports_negative_prompt.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

from conftest import _make_args, _make_plan
from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.image_runner import run_batch


_BASE_CONFIG = {
    "sizes": {"2:3": {"m": {"width": 512, "height": 512}}},
    "generation": {"seed_min": 1, "seed_max": 100},
    "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4},
}


def _config_with_preset(family, supports_negative):
    """Return config with a model_presets entry for the given family."""
    cfg = dict(_BASE_CONFIG)
    cfg["model_presets"] = {
        family: {"supports_negative_prompt": supports_negative},
    }
    return cfg


def _success_workflow():
    from zvisiongenerator.core.workflow import GenerationWorkflow

    stage = MagicMock(return_value=StageOutcome.success)
    return GenerationWorkflow(name="test", stages=[stage])


def _run(family, supports_negative, prompts, size=None):
    """Run a batch whose single stage records its requests; return those requests and the negative-prompt warnings."""
    from zvisiongenerator.utils.image_model_detect import ImageModelInfo

    wf = _success_workflow()
    config = _config_with_preset(family, supports_negative)
    info = ImageModelInfo(family=family, is_distilled=False, size=size)

    with patch("zvisiongenerator.image_runner.build_workflow", return_value=wf):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_batch(MagicMock(), MagicMock(spec=[]), prompts, config, _make_args(), model_info=info, plan=_make_plan(prompts, _make_args(), config=config))

    requests = [call.args[0] for call in wf.stages[0].call_args_list]
    assert all(isinstance(request, ImageGenerationRequest) for request in requests)
    return requests, [warning for warning in caught if "negative prompt" in str(warning.message).lower()]


def test_a_negative_prompt_reaches_a_model_that_supports_it():
    requests, negative_warnings = _run("zimage", True, {"set1": [("a cat", "ugly, blurry")]})

    assert [request.negative_prompt for request in requests] == ["ugly, blurry"]
    assert negative_warnings == []


def test_unsupported_negative_prompts_are_dropped_with_a_single_warning():
    prompts = {
        "set1": [("a city skyline", "blurry, dull"), ("a forest path", "washed out")],
        "set2": [("a harbor at dawn", "low contrast")],
    }

    requests, negative_warnings = _run("ideogram4", False, prompts)

    assert len(negative_warnings) == 1
    assert [request.negative_prompt for request in requests] == [None, None, None]


def test_no_warning_when_no_negatives():
    """No warning if no negative prompts are provided for unsupported models."""
    requests, negative_warnings = _run("flux2_klein", False, {"set1": [("a cat", None)]}, size="4b")

    assert len(requests) == 1
    assert negative_warnings == []
