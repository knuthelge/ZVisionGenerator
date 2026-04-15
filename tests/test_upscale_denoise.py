"""Tests for the upscale-denoise config-driven default resolution.

Verifies:
- 2x upscale gets 0.3 from config when --upscale-denoise is not specified
- 4x upscale gets 0.4 from config when --upscale-denoise is not specified
- Explicit --upscale-denoise value always wins regardless of upscale factor
- runner.run_batch() resolves denoise from config into GenerationRequest
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from conftest import _make_args


@pytest.fixture()
def config():
    """Synthetic config with denoise values — not coupled to shipped defaults."""
    return {
        "sizes": {"2:3": {"m": {"width": 512, "height": 512}}},
        "generation": {"seed_min": 4, "seed_max": 2**32 - 1},
        "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4},
        "upscale": {"default_denoise_2x": 0.25, "default_denoise_4x": 0.45},
        "schedulers": {},
        "model_presets": {"zimage": {"supports_negative_prompt": True}},
    }


class TestCliDefault:
    """CLI parser must default --upscale-denoise to None."""

    def test_default_is_none(self):
        from zvisiongenerator.image_cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["-m", "model"])
        assert args.upscale_denoise is None


class TestRunnerResolvesDenoise:
    """run_batch() must resolve upscale_denoise from config when not explicit."""

    @pytest.fixture()
    def mock_args(self):
        """Minimal argparse.Namespace mirroring what cli.main() passes."""
        return _make_args(
            model="models/fake",
            size="m",
            steps=10,
            guidance=0.5,
            upscale=2,
            upscale_steps=5,
            upscale_sharpen=True,
        )

    def test_runner_resolves_2x_denoise_from_config(self, mock_args, config):
        from zvisiongenerator.core.image_types import ImageGenerationRequest
        from zvisiongenerator.image_runner import run_batch
        from zvisiongenerator.utils.image_model_detect import ImageModelInfo

        mock_backend = MagicMock()
        mock_backend.name = "mock"
        model_info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        prompts_data = {"test_set": [("a cat", None)]}
        mock_args.upscale = 2
        mock_args.upscale_denoise = None

        captured_requests: list[ImageGenerationRequest] = []

        def capture_workflow_run(request, artifacts):
            captured_requests.append(request)
            from zvisiongenerator.core.types import StageOutcome

            return StageOutcome.success

        mock_workflow = MagicMock()
        mock_workflow.run.side_effect = capture_workflow_run

        with patch("zvisiongenerator.image_runner.build_workflow", return_value=mock_workflow):
            run_batch(mock_backend, "fake_model", prompts_data, config, mock_args, model_info=model_info)

        assert len(captured_requests) == 1
        assert captured_requests[0].upscale_denoise == 0.25

    def test_runner_resolves_4x_denoise_from_config(self, mock_args, config):
        from zvisiongenerator.core.image_types import ImageGenerationRequest
        from zvisiongenerator.image_runner import run_batch
        from zvisiongenerator.utils.image_model_detect import ImageModelInfo

        mock_backend = MagicMock()
        mock_backend.name = "mock"
        model_info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        prompts_data = {"test_set": [("a cat", None)]}
        mock_args.upscale = 4
        mock_args.upscale_denoise = None

        captured_requests: list[ImageGenerationRequest] = []

        def capture_workflow_run(request, artifacts):
            captured_requests.append(request)
            from zvisiongenerator.core.types import StageOutcome

            return StageOutcome.success

        mock_workflow = MagicMock()
        mock_workflow.run.side_effect = capture_workflow_run

        with patch("zvisiongenerator.image_runner.build_workflow", return_value=mock_workflow):
            run_batch(mock_backend, "fake_model", prompts_data, config, mock_args, model_info=model_info)

        assert len(captured_requests) == 1
        assert captured_requests[0].upscale_denoise == 0.45

    def test_runner_explicit_denoise_wins(self, mock_args, config):
        from zvisiongenerator.core.image_types import ImageGenerationRequest
        from zvisiongenerator.image_runner import run_batch
        from zvisiongenerator.utils.image_model_detect import ImageModelInfo

        mock_backend = MagicMock()
        mock_backend.name = "mock"
        model_info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
        prompts_data = {"test_set": [("a cat", None)]}
        mock_args.upscale = 2
        mock_args.upscale_denoise = 0.7

        captured_requests: list[ImageGenerationRequest] = []

        def capture_workflow_run(request, artifacts):
            captured_requests.append(request)
            from zvisiongenerator.core.types import StageOutcome

            return StageOutcome.success

        mock_workflow = MagicMock()
        mock_workflow.run.side_effect = capture_workflow_run

        with patch("zvisiongenerator.image_runner.build_workflow", return_value=mock_workflow):
            run_batch(mock_backend, "fake_model", prompts_data, config, mock_args, model_info=model_info)

        assert len(captured_requests) == 1
        assert captured_requests[0].upscale_denoise == 0.7
