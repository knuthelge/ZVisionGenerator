"""Tests for negative prompt handling in batch generation.

Ensures effective_negative is computed before use, and that negative prompts
are correctly preserved or suppressed based on supports_negative_prompt.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

from conftest import _make_args
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


class TestNegativePromptSupported:
    """Model supports negative prompts (e.g., zimage family)."""

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_no_crash_with_negative_prompt(self, mock_get_wf):
        """Regression: effective_negative must be assigned before _display_request."""
        wf = _success_workflow()
        mock_get_wf.return_value = wf

        from zvisiongenerator.utils.image_model_detect import ImageModelInfo

        model_info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        prompts = {"set1": [("a cat", "ugly, blurry")]}
        config = _config_with_preset("zimage", supports_negative=True)

        backend = MagicMock()
        model = MagicMock(spec=[])

        # Should NOT raise UnboundLocalError
        run_batch(backend, model, prompts, config, _make_args(), model_info=model_info)
        wf.stages[0].assert_called_once()

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_negative_prompt_preserved(self, mock_get_wf):
        """When supports_negative_prompt=True, negative prompt flows through."""
        wf = _success_workflow()
        mock_get_wf.return_value = wf

        from zvisiongenerator.utils.image_model_detect import ImageModelInfo

        model_info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        prompts = {"set1": [("a cat", "ugly, blurry")]}
        config = _config_with_preset("zimage", supports_negative=True)

        backend = MagicMock()
        model = MagicMock(spec=[])

        run_batch(backend, model, prompts, config, _make_args(), model_info=model_info)

        # The workflow stage receives a GenerationRequest — check the negative_prompt
        stage_call = wf.stages[0].call_args
        request = stage_call[0][0]  # first positional arg
        assert isinstance(request, ImageGenerationRequest)
        assert request.negative_prompt == "ugly, blurry"


class TestNegativePromptUnsupported:
    """Model does NOT support negative prompts (e.g., flux2 family)."""

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_negative_prompt_suppressed_with_warning(self, mock_get_wf):
        """When supports_negative_prompt=False, negative prompt is suppressed."""
        wf = _success_workflow()
        mock_get_wf.return_value = wf

        from zvisiongenerator.utils.image_model_detect import ImageModelInfo

        model_info = ImageModelInfo(family="flux2_klein", is_distilled=True, size="4b")
        prompts = {"set1": [("a cat", "ugly, blurry")]}
        config = _config_with_preset("flux2_klein", supports_negative=False)

        backend = MagicMock()
        model = MagicMock(spec=[])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_batch(backend, model, prompts, config, _make_args(), model_info=model_info)
            neg_warnings = [x for x in w if "negative prompt" in str(x.message).lower()]
            assert len(neg_warnings) >= 1

        # The workflow stage should receive None for negative_prompt
        stage_call = wf.stages[0].call_args
        request = stage_call[0][0]
        assert isinstance(request, ImageGenerationRequest)
        assert request.negative_prompt is None

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_no_warning_when_no_negatives(self, mock_get_wf):
        """No warning if no negative prompts are provided for unsupported models."""
        wf = _success_workflow()
        mock_get_wf.return_value = wf

        from zvisiongenerator.utils.image_model_detect import ImageModelInfo

        model_info = ImageModelInfo(family="flux2_klein", is_distilled=True, size="4b")
        prompts = {"set1": [("a cat", None)]}
        config = _config_with_preset("flux2_klein", supports_negative=False)

        backend = MagicMock()
        model = MagicMock(spec=[])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_batch(backend, model, prompts, config, _make_args(), model_info=model_info)
            neg_warnings = [x for x in w if "negative prompt" in str(x.message).lower()]
            assert len(neg_warnings) == 0
