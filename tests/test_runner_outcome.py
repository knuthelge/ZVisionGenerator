"""Tests that runner.run_batch() handles StageOutcome values from workflows."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

from conftest import _make_args
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.image_runner import run_batch
from zvisiongenerator.utils.image_model_detect import ImageModelInfo

_MODEL_INFO = ImageModelInfo(family="zimage", is_distilled=False, size=None)


_CONFIG = {
    "sizes": {"2:3": {"m": {"width": 512, "height": 512}}},
    "generation": {"seed_min": 1, "seed_max": 100},
    "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4},
    "model_presets": {"zimage": {"supports_negative_prompt": True}},
}


def _prompts(n=1):
    return {"set1": [("a photo of a cat", None)] * n}


def _mock_workflow(outcome: StageOutcome) -> GenerationWorkflow:
    """Create a workflow whose single stage always returns *outcome*."""
    stage = MagicMock(return_value=outcome)
    return GenerationWorkflow(name="test", stages=[stage])


class TestRunnerOutcome:
    """Verify run_batch() reacts correctly to each StageOutcome."""

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_success_continues_normally(self, mock_build_wf):
        wf = _mock_workflow(StageOutcome.success)
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])  # no _model_info attr

        run_batch(backend, model, _prompts(), _CONFIG, _make_args(), model_info=_MODEL_INFO)
        wf.stages[0].assert_called_once()

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_skipped_warns_and_continues(self, mock_build_wf):
        wf = _mock_workflow(StageOutcome.skipped)
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_batch(backend, model, _prompts(2), _CONFIG, _make_args(), model_info=_MODEL_INFO)
            skipped_warnings = [x for x in w if "skipped" in str(x.message).lower()]
            assert len(skipped_warnings) >= 1

        # Both prompts should still have been attempted
        assert wf.stages[0].call_count == 2

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_failed_warns_and_continues(self, mock_build_wf):
        wf = _mock_workflow(StageOutcome.failed)
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_batch(backend, model, _prompts(2), _CONFIG, _make_args(), model_info=_MODEL_INFO)
            failed_warnings = [x for x in w if "failed" in str(x.message).lower()]
            assert len(failed_warnings) >= 1

        assert wf.stages[0].call_count == 2

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_retry_warns_and_continues(self, mock_build_wf):
        wf = _mock_workflow(StageOutcome.retry)
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_batch(backend, model, _prompts(2), _CONFIG, _make_args(), model_info=_MODEL_INFO)
            retry_warnings = [x for x in w if "retry" in str(x.message).lower()]
            assert len(retry_warnings) >= 1
            # Should include a "failed after N retries" warning per prompt
            exceeded_warnings = [x for x in w if "failed after" in str(x.message).lower()]
            assert len(exceeded_warnings) == 2

        # 4 attempts per prompt (1 initial + 3 retries) × 2 prompts = 8
        assert wf.stages[0].call_count == 8

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_repeat_regenerates_seed(self, mock_build_wf):
        """Pressing 'r' (repeat) should generate a new seed for the second run."""
        call_count = 0
        seeds_seen: list[int] = []

        def _capture_and_succeed(request, artifacts):
            nonlocal call_count
            seeds_seen.append(request.seed)
            call_count += 1
            return StageOutcome.success

        stage = MagicMock(side_effect=_capture_and_succeed)
        wf = GenerationWorkflow(name="test", stages=[stage])
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])

        # Mock random.randint to return controlled, distinct values
        with patch("zvisiongenerator.image_runner.SkipSignal") as MockSkip, patch("zvisiongenerator.image_runner.random.randint", side_effect=[100, 200]):
            skip_inst = MockSkip.return_value
            skip_inst.consume.side_effect = ["repeat", "skip"]
            skip_inst.reset = MagicMock()
            skip_inst.start = MagicMock()
            skip_inst.stop = MagicMock()
            skip_inst.wait_for_key = MagicMock()

            run_batch(backend, model, _prompts(1), _CONFIG, _make_args(seed=None), model_info=_MODEL_INFO)

        assert call_count == 2
        assert len(seeds_seen) == 2
        assert seeds_seen[0] == 100
        assert seeds_seen[1] == 200

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_repeat_preserves_explicit_seed(self, mock_build_wf):
        """When --seed is explicitly set, repeat should reuse that fixed seed."""
        seeds_seen: list[int] = []

        def _capture_and_succeed(request, artifacts):
            seeds_seen.append(request.seed)
            return StageOutcome.success

        stage = MagicMock(side_effect=_capture_and_succeed)
        wf = GenerationWorkflow(name="test", stages=[stage])
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])

        with patch("zvisiongenerator.image_runner.SkipSignal") as MockSkip:
            skip_inst = MockSkip.return_value
            skip_inst.consume.side_effect = ["repeat", "skip"]
            skip_inst.reset = MagicMock()
            skip_inst.start = MagicMock()
            skip_inst.stop = MagicMock()
            skip_inst.wait_for_key = MagicMock()

            run_batch(backend, model, _prompts(1), _CONFIG, _make_args(seed=42), model_info=_MODEL_INFO)

        assert len(seeds_seen) == 2
        assert seeds_seen[0] == 42
        assert seeds_seen[1] == 42

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_quit_during_generation_ends_batch(self, mock_build_wf):
        """Pressing 'q' during generation (skipped outcome) should end the batch immediately."""
        wf = _mock_workflow(StageOutcome.skipped)
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])

        with patch("zvisiongenerator.image_runner.SkipSignal") as MockSkip:
            skip_inst = MockSkip.return_value
            # consume() returns "quit" — simulating user pressed 'q' during generation
            skip_inst.consume.return_value = "quit"
            skip_inst.reset = MagicMock()
            skip_inst.start = MagicMock()
            skip_inst.stop = MagicMock()
            skip_inst.check = MagicMock(return_value=True)

            # With 3 prompts, only the first should run before quit ends the batch
            run_batch(backend, model, _prompts(3), _CONFIG, _make_args(), model_info=_MODEL_INFO)

        assert wf.stages[0].call_count == 1


# ---------------------------------------------------------------------------
# Preset size drift warning with upscale
# ---------------------------------------------------------------------------


class TestPresetSizeDriftWarning:
    """run_batch should warn when preset size drifts under upscale round-trip."""

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_warns_on_drift(self, mock_wf):
        """Preset 'm' with 4x upscale: 1440//4=360, _round_to_16(360)=368, 368*4=1472 != 1440."""
        stage = MagicMock(return_value=StageOutcome.success)
        mock_wf.return_value = GenerationWorkflow(name="t", stages=[stage])

        config = {
            "sizes": {"2:3": {"m": {"width": 1440, "height": 768}}},
            "generation": {"seed_min": 1, "seed_max": 100},
            "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4},
            "model_presets": {"zimage": {"supports_negative_prompt": True}},
        }
        args = _make_args(
            size="m",
            upscale=4,
            upscale_denoise=0.4,
            upscale_steps=2,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_batch(MagicMock(), MagicMock(spec=[]), {"s": [("cat", None)]}, config, args, model_info=_MODEL_INFO)

        drift_warnings = [x for x in w if "drifts" in str(x.message)]
        assert len(drift_warnings) >= 1
        assert "1440" in str(drift_warnings[0].message)

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_no_warning_when_aligned(self, mock_wf):
        """Preset with dimensions that survive upscale round-trip should emit no drift warning."""
        stage = MagicMock(return_value=StageOutcome.success)
        mock_wf.return_value = GenerationWorkflow(name="t", stages=[stage])

        config = {
            "sizes": {"2:3": {"a": {"width": 1024, "height": 768}}},
            "generation": {"seed_min": 1, "seed_max": 100},
            "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4},
            "model_presets": {"zimage": {"supports_negative_prompt": True}},
        }
        args = _make_args(
            size="a",
            upscale=4,
            upscale_denoise=0.4,
            upscale_steps=2,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_batch(MagicMock(), MagicMock(spec=[]), {"s": [("cat", None)]}, config, args, model_info=_MODEL_INFO)

        drift_warnings = [x for x in w if "drifts" in str(x.message)]
        assert len(drift_warnings) == 0

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_no_warning_with_explicit_dims(self, mock_wf):
        """Explicit --width/--height should skip drift check even with upscale."""
        stage = MagicMock(return_value=StageOutcome.success)
        mock_wf.return_value = GenerationWorkflow(name="t", stages=[stage])

        config = {
            "sizes": {"2:3": {"m": {"width": 1440, "height": 768}}},
            "generation": {"seed_min": 1, "seed_max": 100},
            "sharpening": {"normal": 0.8, "upscaled": 1.2, "pre_upscale": 0.4},
            "model_presets": {"zimage": {"supports_negative_prompt": True}},
        }
        args = _make_args(
            size="m",
            width=1024,
            height=768,
            upscale=4,
            upscale_denoise=0.4,
            upscale_steps=2,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_batch(MagicMock(), MagicMock(spec=[]), {"s": [("cat", None)]}, config, args, model_info=_MODEL_INFO)

        drift_warnings = [x for x in w if "drifts" in str(x.message)]
        assert len(drift_warnings) == 0


# ---------------------------------------------------------------------------
# Amount-propagation: verify runner resolves flag values into GenerationRequest
# ---------------------------------------------------------------------------


class TestAmountPropagation:
    """Verify runner correctly translates CLI flag values to GenerationRequest fields."""

    def _capture_request(self, args_overrides: dict):
        """Run a single-prompt batch capturing the GenerationRequest passed to workflow.run()."""
        captured = {}

        def _capture_stage(request, artifacts):
            captured["request"] = request
            return StageOutcome.success

        wf = GenerationWorkflow(name="test", stages=[_capture_stage])
        with patch("zvisiongenerator.image_runner.build_workflow", return_value=wf):
            run_batch(
                MagicMock(),
                MagicMock(spec=[]),
                _prompts(1),
                _CONFIG,
                _make_args(**args_overrides),
                model_info=_MODEL_INFO,
            )
        return captured["request"]

    # -- sharpen --

    def test_sharpen_float_sets_override(self):
        req = self._capture_request({"sharpen": 0.6})
        assert req.sharpen is True
        assert req.sharpen_amount_override == 0.6

    def test_sharpen_true_no_override(self):
        req = self._capture_request({"sharpen": True})
        assert req.sharpen is True
        assert req.sharpen_amount_override is None

    def test_sharpen_false_disables(self):
        req = self._capture_request({"sharpen": False})
        assert req.sharpen is False

    def test_sharpen_zero_is_not_false(self):
        req = self._capture_request({"sharpen": 0.0})
        assert req.sharpen is True
        assert req.sharpen_amount_override == 0.0

    # -- contrast --

    def test_contrast_float_sets_amount(self):
        req = self._capture_request({"contrast": 1.3})
        assert req.contrast is True
        assert req.contrast_amount == 1.3

    def test_contrast_true_uses_config_default(self):
        config = {**_CONFIG, "contrast": {"default_amount": 1.5}}
        captured = {}

        def _capture_stage(request, artifacts):
            captured["request"] = request
            return StageOutcome.success

        wf = GenerationWorkflow(name="test", stages=[_capture_stage])
        with patch("zvisiongenerator.image_runner.build_workflow", return_value=wf):
            run_batch(
                MagicMock(),
                MagicMock(spec=[]),
                _prompts(1),
                config,
                _make_args(contrast=True),
                model_info=_MODEL_INFO,
            )
        req = captured["request"]
        assert req.contrast is True
        assert req.contrast_amount == 1.5

    def test_contrast_false_disables(self):
        req = self._capture_request({"contrast": False})
        assert req.contrast is False

    def test_contrast_zero_is_not_false(self):
        req = self._capture_request({"contrast": 0.0})
        assert req.contrast is True
        assert req.contrast_amount == 0.0

    # -- saturation --

    def test_saturation_float_sets_amount(self):
        req = self._capture_request({"saturation": 1.2})
        assert req.saturation is True
        assert req.saturation_amount == 1.2

    def test_saturation_true_uses_config_default(self):
        config = {**_CONFIG, "saturation": {"default_amount": 0.8}}
        captured = {}

        def _capture_stage(request, artifacts):
            captured["request"] = request
            return StageOutcome.success

        wf = GenerationWorkflow(name="test", stages=[_capture_stage])
        with patch("zvisiongenerator.image_runner.build_workflow", return_value=wf):
            run_batch(
                MagicMock(),
                MagicMock(spec=[]),
                _prompts(1),
                config,
                _make_args(saturation=True),
                model_info=_MODEL_INFO,
            )
        req = captured["request"]
        assert req.saturation is True
        assert req.saturation_amount == 0.8

    def test_saturation_false_disables(self):
        req = self._capture_request({"saturation": False})
        assert req.saturation is False

    def test_saturation_zero_is_not_false(self):
        req = self._capture_request({"saturation": 0.0})
        assert req.saturation is True
        assert req.saturation_amount == 0.0
