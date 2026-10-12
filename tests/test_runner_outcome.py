"""Tests that runner.run_batch() handles StageOutcome values from workflows."""

from __future__ import annotations

import warnings
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from conftest import _make_args, _make_plan
from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.job_plan import EnhanceStatus, JobPlan
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.image_runner import run_batch
from zvisiongenerator.utils.image_model_detect import ImageModelInfo
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings

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
    def test_success_emits_generation_finished_with_shared_payload(self, mock_build_wf):
        """Successful image generations should emit the shared generation_finished payload."""
        events: list[dict[str, object]] = []

        def _capture_and_succeed(request, artifacts):
            artifacts.filename = "image.png"
            artifacts.filepath = "/tmp/image.png"
            return StageOutcome.success

        stage = MagicMock(side_effect=_capture_and_succeed)
        mock_build_wf.return_value = GenerationWorkflow(name="test", stages=[stage])

        backend = MagicMock()
        model = MagicMock(spec=[])

        run_batch(
            backend,
            model,
            _prompts(),
            _CONFIG,
            _make_args(seed=42),
            model_info=_MODEL_INFO,
            progress_callback=events.append,
            plan=_make_plan(_prompts(), _make_args(seed=42), config=_CONFIG),
        )

        finished_events = [event for event in events if event["type"] == "generation_finished"]

        assert len(finished_events) == 1
        assert finished_events[0]["mode"] == "image"
        assert finished_events[0]["status"] == "success"
        assert finished_events[0]["run_index"] == 0
        assert finished_events[0]["total_runs"] == 1
        assert finished_events[0]["ran_iterations"] == 1
        assert finished_events[0]["total_iterations"] == 1
        assert finished_events[0]["set_name"] == "set1"
        assert finished_events[0]["prompt_index"] == 0
        assert finished_events[0]["total_prompts"] == 1
        assert finished_events[0]["prompt"] == "a photo of a cat"
        assert finished_events[0]["seed"] == 42
        assert finished_events[0]["filename"] == "image.png"
        assert finished_events[0]["output_path"] == "/tmp/image.png"
        assert isinstance(finished_events[0]["generation_time"], float)

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_success_continues_normally(self, mock_build_wf):
        wf = _mock_workflow(StageOutcome.success)
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])  # no _model_info attr

        run_batch(backend, model, _prompts(), _CONFIG, _make_args(), model_info=_MODEL_INFO, plan=_make_plan(_prompts(), _make_args(), config=_CONFIG))
        wf.stages[0].assert_called_once()

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_request_json_prompt_defaults_false(self, mock_build_wf):
        wf = _mock_workflow(StageOutcome.success)
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])

        run_batch(backend, model, _prompts(), _CONFIG, _make_args(), model_info=_MODEL_INFO, plan=_make_plan(_prompts(), _make_args(), config=_CONFIG))

        request = wf.stages[0].call_args[0][0]
        assert request.json_prompt is False

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_request_threads_json_prompt_from_args(self, mock_build_wf):
        wf = _mock_workflow(StageOutcome.success)
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])

        run_batch(backend, model, _prompts(), _CONFIG, _make_args(json_prompt_enabled=True), model_info=_MODEL_INFO, plan=_make_plan(_prompts(), _make_args(json_prompt_enabled=True), config=_CONFIG))

        request = wf.stages[0].call_args[0][0]
        assert request.json_prompt is True

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_request_threads_first_sigma_from_args(self, mock_build_wf):
        wf = _mock_workflow(StageOutcome.success)
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])

        run_batch(backend, model, _prompts(), _CONFIG, _make_args(first_sigma=1.005), model_info=_MODEL_INFO, plan=_make_plan(_prompts(), _make_args(first_sigma=1.005), config=_CONFIG))

        request = wf.stages[0].call_args[0][0]
        assert request.first_sigma == 1.005

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_request_first_sigma_defaults_to_none_when_args_omit_it(self, mock_build_wf):
        wf = _mock_workflow(StageOutcome.success)
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])

        run_batch(backend, model, _prompts(), _CONFIG, _make_args(), model_info=_MODEL_INFO, plan=_make_plan(_prompts(), _make_args(), config=_CONFIG))

        request = wf.stages[0].call_args[0][0]
        assert request.first_sigma is None

    def test_image_generation_request_first_sigma_default_is_none(self):
        request = ImageGenerationRequest(backend=MagicMock(), model=MagicMock(), prompt="prompt")

        assert request.first_sigma is None

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_skipped_warns_and_continues(self, mock_build_wf):
        wf = _mock_workflow(StageOutcome.skipped)
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_batch(backend, model, _prompts(2), _CONFIG, _make_args(), model_info=_MODEL_INFO, plan=_make_plan(_prompts(2), _make_args(), config=_CONFIG))
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
            run_batch(backend, model, _prompts(2), _CONFIG, _make_args(), model_info=_MODEL_INFO, plan=_make_plan(_prompts(2), _make_args(), config=_CONFIG))
            failed_warnings = [x for x in w if "failed" in str(x.message).lower()]
            assert len(failed_warnings) >= 1

        assert wf.stages[0].call_count == 2

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_all_failed_batch_emits_failed_terminal_event(self, mock_build_wf):
        wf = _mock_workflow(StageOutcome.failed)
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])
        events: list[dict[str, object]] = []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_batch(
                backend,
                model,
                _prompts(),
                _CONFIG,
                _make_args(seed=42),
                model_info=_MODEL_INFO,
                progress_callback=events.append,
                plan=_make_plan(_prompts(), _make_args(seed=42), config=_CONFIG),
            )

        failed_warnings = [warning for warning in w if "failed" in str(warning.message).lower()]
        finished_events = [event for event in events if event["type"] == "generation_finished"]
        batch_failed_events = [event for event in events if event["type"] == "batch_failed"]

        assert len(failed_warnings) >= 1
        assert len(finished_events) == 1
        assert finished_events[0]["type"] == "generation_finished"
        assert finished_events[0]["mode"] == "image"
        assert finished_events[0]["run_index"] == 0
        assert finished_events[0]["total_runs"] == 1
        assert finished_events[0]["ran_iterations"] == 1
        assert finished_events[0]["total_iterations"] == 1
        assert finished_events[0]["set_name"] == "set1"
        assert finished_events[0]["prompt_index"] == 0
        assert finished_events[0]["total_prompts"] == 1
        assert finished_events[0]["prompt"] == "a photo of a cat"
        assert finished_events[0]["seed"] == 42
        assert finished_events[0]["status"] == "failed"
        assert isinstance(finished_events[0]["filename"], str)
        assert isinstance(finished_events[0]["generation_time"], float)
        assert "output_path" not in finished_events[0]
        assert len(batch_failed_events) == 1
        assert not [event for event in events if event["type"] == "batch_completed"]
        assert batch_failed_events[0] == {
            "type": "batch_failed",
            "mode": "image",
            "completed_iterations": 1,
            "total_iterations": 1,
        }

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_retry_warns_and_continues(self, mock_build_wf):
        wf = _mock_workflow(StageOutcome.retry)
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_batch(backend, model, _prompts(2), _CONFIG, _make_args(), model_info=_MODEL_INFO, plan=_make_plan(_prompts(2), _make_args(), config=_CONFIG))
            retry_warnings = [x for x in w if "retry" in str(x.message).lower()]
            assert len(retry_warnings) >= 1
            # Should include a "failed after N retries" warning per prompt
            exceeded_warnings = [x for x in w if "failed after" in str(x.message).lower()]
            assert len(exceeded_warnings) == 2

        # 4 attempts per prompt (1 initial + 3 retries) × 2 prompts = 8
        assert wf.stages[0].call_count == 8

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_retry_exhaustion_emits_failed_generation_and_batch_failed(self, mock_build_wf):
        wf = _mock_workflow(StageOutcome.retry)
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])
        events: list[dict[str, object]] = []

        plan = _make_plan(_prompts(), _make_args(seed=42), config=_CONFIG)
        with warnings.catch_warnings(record=True) as w, patch("zvisiongenerator.image_runner.random.randint", side_effect=[10, 20, 30]):
            warnings.simplefilter("always")
            run_batch(
                backend,
                model,
                _prompts(),
                _CONFIG,
                _make_args(seed=42),
                model_info=_MODEL_INFO,
                progress_callback=events.append,
                plan=plan,
            )

        retry_exhausted_warnings = [warning for warning in w if "failed after" in str(warning.message).lower()]
        finished_events = [event for event in events if event["type"] == "generation_finished"]
        batch_failed_events = [event for event in events if event["type"] == "batch_failed"]

        assert len(retry_exhausted_warnings) == 1
        assert wf.stages[0].call_count == 4
        assert len(finished_events) == 1
        assert finished_events[0]["status"] == "failed"
        # Retries draw a random seed even when the job's seed is fixed.
        assert finished_events[0]["seed"] == 30
        assert isinstance(finished_events[0]["filename"], str)
        assert isinstance(finished_events[0]["generation_time"], float)
        assert "output_path" not in finished_events[0]
        assert len(batch_failed_events) == 1
        assert batch_failed_events[0] == {
            "type": "batch_failed",
            "mode": "image",
            "completed_iterations": 1,
            "total_iterations": 1,
        }
        assert not [event for event in events if event["type"] == "batch_completed"]

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_repeat_regenerates_seed(self, mock_build_wf):
        """Pressing 'r' (repeat) should generate a new seed for the second run and keep the planned text."""
        call_count = 0
        seeds_seen: list[int] = []
        texts_seen: list[tuple[str | None, str | None]] = []

        def _capture_and_succeed(request, artifacts):
            nonlocal call_count
            seeds_seen.append(request.seed)
            texts_seen.append((request.resolved_prompt, request.enhanced_prompt))
            call_count += 1
            return StageOutcome.success

        stage = MagicMock(side_effect=_capture_and_succeed)
        wf = GenerationWorkflow(name="test", stages=[stage])
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])

        plan = _make_plan(_prompts(1), _make_args(seed=None), config=_CONFIG)
        # Mock random.randint to return a controlled value for the repeat
        with patch("zvisiongenerator.image_runner.SkipSignal") as MockSkip, patch("zvisiongenerator.image_runner.random.randint", side_effect=[200]):
            skip_inst = MockSkip.return_value
            # None: nothing queued before each generation; then the post-generation action.
            skip_inst.consume.side_effect = [None, "repeat", None, "skip"]
            skip_inst.reset = MagicMock()
            skip_inst.start = MagicMock()
            skip_inst.stop = MagicMock()
            skip_inst.wait_for_key = MagicMock()

            run_batch(backend, model, _prompts(1), _CONFIG, _make_args(seed=None), model_info=_MODEL_INFO, plan=plan)

        assert call_count == 2
        assert seeds_seen == [plan.iterations[0].seed, 200]
        assert texts_seen == [("a photo of a cat", None)] * 2

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_repeat_draws_random_seed_even_with_explicit_seed(self, mock_build_wf):
        """Repeat always draws a new random seed, even when --seed is set."""
        seeds_seen: list[int] = []

        def _capture_and_succeed(request, artifacts):
            seeds_seen.append(request.seed)
            return StageOutcome.success

        stage = MagicMock(side_effect=_capture_and_succeed)
        wf = GenerationWorkflow(name="test", stages=[stage])
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])

        with patch("zvisiongenerator.image_runner.SkipSignal") as MockSkip, patch("zvisiongenerator.image_runner.random.randint", side_effect=[77]) as randint:
            skip_inst = MockSkip.return_value
            # None: nothing queued before each generation; then the post-generation action.
            skip_inst.consume.side_effect = [None, "repeat", None, "skip"]
            skip_inst.reset = MagicMock()
            skip_inst.start = MagicMock()
            skip_inst.stop = MagicMock()
            skip_inst.wait_for_key = MagicMock()

            run_batch(backend, model, _prompts(1), _CONFIG, _make_args(seed=42), model_info=_MODEL_INFO, plan=_make_plan(_prompts(1), _make_args(seed=42), config=_CONFIG))

        assert seeds_seen == [42, 77]
        randint.assert_called_once_with(1, 100)

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_quit_during_generation_ends_batch(self, mock_build_wf):
        """Pressing 'q' during generation (skipped outcome) should end the batch immediately."""
        wf = _mock_workflow(StageOutcome.skipped)
        mock_build_wf.return_value = wf

        backend = MagicMock()
        model = MagicMock(spec=[])

        with patch("zvisiongenerator.image_runner.SkipSignal") as MockSkip:
            skip_inst = MockSkip.return_value
            # Nothing queued before the first generation; consume() then returns "quit" — user pressed 'q' during it
            skip_inst.consume.side_effect = [None, "quit"]
            skip_inst.reset = MagicMock()
            skip_inst.start = MagicMock()
            skip_inst.stop = MagicMock()
            skip_inst.check = MagicMock(return_value=True)

            # With 3 prompts, only the first should run before quit ends the batch
            run_batch(backend, model, _prompts(3), _CONFIG, _make_args(), model_info=_MODEL_INFO, plan=_make_plan(_prompts(3), _make_args(), config=_CONFIG))

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
            run_batch(MagicMock(), MagicMock(spec=[]), {"s": [("cat", None)]}, config, args, model_info=_MODEL_INFO, plan=_make_plan({"s": [("cat", None)]}, args, config=config))

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
            run_batch(MagicMock(), MagicMock(spec=[]), {"s": [("cat", None)]}, config, args, model_info=_MODEL_INFO, plan=_make_plan({"s": [("cat", None)]}, args, config=config))

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
            run_batch(MagicMock(), MagicMock(spec=[]), {"s": [("cat", None)]}, config, args, model_info=_MODEL_INFO, plan=_make_plan({"s": [("cat", None)]}, args, config=config))

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
                plan=_make_plan(_prompts(1), _make_args(**args_overrides), config=_CONFIG),
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
                plan=_make_plan(_prompts(1), _make_args(contrast=True), config=config),
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
                plan=_make_plan(_prompts(1), _make_args(saturation=True), config=config),
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


class TestProgressCallbacks:
    """Verify run_batch forwards low-level generation progress through the structured callback API."""

    def test_step_progress_includes_image_generation_context(self):
        events: list[dict[str, object]] = []

        def _emit_step_then_succeed(request, artifacts):
            assert request.step_callback is not None
            request.step_callback({"current_step": 2, "total_steps": request.steps})
            return StageOutcome.success

        workflow = GenerationWorkflow(name="test", stages=[_emit_step_then_succeed])

        with patch("zvisiongenerator.image_runner.build_workflow", return_value=workflow):
            run_batch(
                MagicMock(),
                MagicMock(spec=[]),
                _prompts(1),
                _CONFIG,
                _make_args(steps=4),
                model_info=_MODEL_INFO,
                progress_callback=events.append,
                plan=_make_plan(_prompts(1), _make_args(steps=4), config=_CONFIG),
            )

        step_events = [event for event in events if event["type"] == "step_progress"]

        assert len(step_events) == 1
        assert step_events[0]["mode"] == "image"
        assert step_events[0]["current_step"] == 2
        assert step_events[0]["total_steps"] == 4
        assert step_events[0]["run_index"] == 0
        assert step_events[0]["total_runs"] == 1
        assert step_events[0]["ran_iterations"] == 1
        assert step_events[0]["total_iterations"] == 1
        assert step_events[0]["set_name"] == "set1"
        assert step_events[0]["prompt_index"] == 0
        assert step_events[0]["total_prompts"] == 1


class TestQueuedControlsBeforeGeneration:
    """Controls queued before a generation starts (e.g. during model load) must not be discarded."""

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_quit_queued_before_batch_cancels_without_generating(self, mock_build_wf):
        stage = MagicMock(return_value=StageOutcome.success)
        mock_build_wf.return_value = GenerationWorkflow(name="test", stages=[stage])
        skip = SkipSignal()
        skip.queue_action("quit")
        events: list[dict[str, object]] = []

        run_batch(
            MagicMock(),
            MagicMock(spec=[]),
            _prompts(3),
            _CONFIG,
            _make_args(),
            model_info=_MODEL_INFO,
            progress_callback=events.append,
            skip_signal=skip,
            plan=_make_plan(_prompts(3), _make_args(), config=_CONFIG),
        )

        stage.assert_not_called()
        event_types = [event["type"] for event in events]
        assert "batch_cancelled" in event_types
        assert "batch_completed" not in event_types

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_stale_skip_queued_before_generation_is_dropped(self, mock_build_wf):
        stage = MagicMock(return_value=StageOutcome.success)
        mock_build_wf.return_value = GenerationWorkflow(name="test", stages=[stage])
        skip = SkipSignal()
        skip.queue_action("skip")

        run_batch(
            MagicMock(),
            MagicMock(spec=[]),
            _prompts(2),
            _CONFIG,
            _make_args(),
            model_info=_MODEL_INFO,
            skip_signal=skip,
            plan=_make_plan(_prompts(2), _make_args(), config=_CONFIG),
        )

        assert stage.call_count == 2


class TestQueuedPauseBeforeGeneration:
    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_pause_queued_before_generation_waits_for_resume(self, mock_build_wf):
        import threading
        import time

        stage = MagicMock(return_value=StageOutcome.success)
        mock_build_wf.return_value = GenerationWorkflow(name="test", stages=[stage])
        skip = SkipSignal()
        skip.queue_action("pause")
        events: list[dict[str, object]] = []

        def _resume_once_waiting():
            while not skip.is_waiting_for_resume():
                time.sleep(0.001)
            skip.resume()

        def _resume_when_paused(event):
            events.append(event)
            if event["type"] == "job_paused":
                assert stage.call_count == 0
                threading.Thread(target=_resume_once_waiting, daemon=True).start()

        run_batch(
            MagicMock(),
            MagicMock(spec=[]),
            _prompts(1),
            _CONFIG,
            _make_args(),
            model_info=_MODEL_INFO,
            progress_callback=_resume_when_paused,
            skip_signal=skip,
            plan=_make_plan(_prompts(1), _make_args(), config=_CONFIG),
        )

        event_types = [event["type"] for event in events]
        assert event_types.index("job_paused") < event_types.index("job_resumed") < event_types.index("generation_finished")
        assert stage.call_count == 1


def _enhanced_plan(prompts, args):
    """Return a plan whose first iteration was enhanced during preflight (other iterations untouched)."""
    plan = _make_plan(prompts, args, config=_CONFIG)
    first = replace(plan.iterations[0], resolved_prompt="a photo of a tabby cat", enhance=EnhanceSettings(), enhanced_prompt="A tabby cat in warm light.", enhance_status=EnhanceStatus.ENHANCED)
    return JobPlan(iterations=(first, *plan.iterations[1:]))


class TestPlanConsumption:
    """run_batch() takes seeds and text from the preflight plan."""

    def test_cancelled_plan_raises(self):
        with pytest.raises(ValueError, match="cancelled"):
            run_batch(MagicMock(), MagicMock(), _prompts(), _CONFIG, _make_args(), model_info=_MODEL_INFO, plan=JobPlan(iterations=(), cancelled=True))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="iterations"):
            run_batch(MagicMock(), MagicMock(), _prompts(2), _CONFIG, _make_args(), model_info=_MODEL_INFO, plan=_make_plan(_prompts(1), _make_args(), config=_CONFIG))

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_order_mismatch_raises(self, mock_build_wf):
        mock_build_wf.return_value = _mock_workflow(StageOutcome.success)
        plan = _make_plan({"other": [("x", None)]}, _make_args(), config=_CONFIG)
        with pytest.raises(RuntimeError, match="out of order"):
            run_batch(MagicMock(), MagicMock(spec=[]), _prompts(), _CONFIG, _make_args(), model_info=_MODEL_INFO, plan=plan)

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_request_carries_planned_seed_and_text(self, mock_build_wf):
        requests: list[ImageGenerationRequest] = []
        mock_build_wf.return_value = GenerationWorkflow(name="t", stages=[MagicMock(side_effect=lambda r, a: requests.append(r) or StageOutcome.success)])
        plan = _enhanced_plan(_prompts(2), _make_args(seed=None))
        run_batch(MagicMock(), MagicMock(spec=[]), _prompts(2), _CONFIG, _make_args(seed=None), model_info=_MODEL_INFO, plan=plan)
        assert [r.seed for r in requests] == [it.seed for it in plan.iterations]
        assert [(r.resolved_prompt, r.enhanced_prompt) for r in requests] == [("a photo of a tabby cat", "A tabby cat in warm light."), ("a photo of a cat", None)]
        assert mock_build_wf.call_args.kwargs == {"enhance": True}

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_workflow_omits_enhance_stage_without_rewrites(self, mock_build_wf):
        mock_build_wf.return_value = _mock_workflow(StageOutcome.success)
        run_batch(MagicMock(), MagicMock(spec=[]), _prompts(), _CONFIG, _make_args(), model_info=_MODEL_INFO, plan=_make_plan(_prompts(), _make_args(), config=_CONFIG))
        assert mock_build_wf.call_args.kwargs == {"enhance": False}

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_prompt_started_reports_enhance_status(self, mock_build_wf):
        mock_build_wf.return_value = _mock_workflow(StageOutcome.success)
        events: list[dict] = []
        run_batch(MagicMock(), MagicMock(spec=[]), _prompts(2), _CONFIG, _make_args(), model_info=_MODEL_INFO, plan=_enhanced_plan(_prompts(2), _make_args()), progress_callback=events.append)
        assert [e["enhance_status"] for e in events if e["type"] == "prompt_started"] == ["enhanced", "off"]

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_retry_keeps_text_and_draws_random_seed(self, mock_build_wf):
        requests: list[ImageGenerationRequest] = []
        outcomes = iter([StageOutcome.retry, StageOutcome.success])
        mock_build_wf.return_value = GenerationWorkflow(name="t", stages=[MagicMock(side_effect=lambda r, a: requests.append(r) or next(outcomes))])
        with warnings.catch_warnings(), patch("zvisiongenerator.image_runner.random.randint", side_effect=[55]):
            warnings.simplefilter("ignore")
            run_batch(MagicMock(), MagicMock(spec=[]), _prompts(), _CONFIG, _make_args(seed=42), model_info=_MODEL_INFO, plan=_enhanced_plan(_prompts(), _make_args(seed=42)))
        assert [r.seed for r in requests] == [42, 55]
        assert {(r.resolved_prompt, r.enhanced_prompt) for r in requests} == {("a photo of a tabby cat", "A tabby cat in warm light.")}

    def test_prompt_enhanced_only_for_enhanced_iterations_on_every_attempt(self):
        """The real pass-through stage emits prompt_enhanced after prompt_started, once per attempt."""
        events: list[dict] = []
        skip = SkipSignal()
        consumed = iter([None, "repeat", None, None, None, None])
        skip.consume = lambda: next(consumed)
        from zvisiongenerator.workflows.image_stages import enhance_prompt_stage, resolve_prompt_stage

        with patch("zvisiongenerator.image_runner.build_workflow") as mock_build_wf:
            mock_build_wf.return_value = GenerationWorkflow(name="t", stages=[resolve_prompt_stage, enhance_prompt_stage])
            run_batch(
                MagicMock(),
                MagicMock(spec=[]),
                _prompts(2),
                _CONFIG,
                _make_args(),
                model_info=_MODEL_INFO,
                plan=_enhanced_plan(_prompts(2), _make_args()),
                progress_callback=events.append,
                skip_signal=skip,
            )
        types = [e["type"] for e in events if e["type"] in ("prompt_started", "prompt_enhanced")]
        assert types == ["prompt_started", "prompt_enhanced", "prompt_enhanced", "prompt_started"]
        assert {e["enhanced_prompt"] for e in events if e["type"] == "prompt_enhanced"} == {"A tabby cat in warm light."}


class TestReviewBeforeUpscale:
    """run_batch builds the review callable and reacts to its decision."""

    @staticmethod
    def _workflow(upscale):
        from PIL import Image

        from zvisiongenerator.workflows.image_stages import review_stage

        def _render(request, artifacts):
            artifacts.image = Image.new("RGB", (8, 8))
            artifacts.filename = "image"
            artifacts.filepath = "/out/image.png"
            return StageOutcome.success

        return GenerationWorkflow(name="test", stages=[_render, review_stage, upscale])

    @staticmethod
    def _run(skip, decide, *, prompts=1, args=None, on_event=None):
        """Run a batch whose reviews are decided by *decide(event)* on each review's job_paused event."""
        events: list[dict[str, object]] = []

        def _callback(event):
            events.append(event)
            if event["type"] == "job_paused" and "review" in event:
                decide(event)
            if on_event is not None:
                on_event(event)

        args = args or _make_args(upscale=2, review=True)
        run_batch(
            MagicMock(),
            MagicMock(spec=[]),
            _prompts(prompts),
            _CONFIG,
            args,
            model_info=_MODEL_INFO,
            progress_callback=_callback,
            skip_signal=skip,
            plan=_make_plan(_prompts(prompts), args, config=_CONFIG),
        )
        return events

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_approve_pauses_with_review_and_preview_then_resumes_and_upscales(self, mock_build_wf):
        upscale = MagicMock(return_value=StageOutcome.success)
        mock_build_wf.return_value = self._workflow(upscale)
        skip = SkipSignal()

        events = self._run(skip, lambda _e: skip.set_review_decision("approve"))

        types = [event["type"] for event in events]
        assert types.index("job_paused") < types.index("job_resumed") < types.index("generation_finished")
        paused = next(event for event in events if event["type"] == "job_paused")
        assert paused["review"] == {"iteration": 1}
        assert paused["preview"].size == (8, 8)
        finished = next(event for event in events if event["type"] == "generation_finished")
        assert finished["status"] == "success"
        upscale.assert_called_once()
        assert types[-1] == "batch_completed"

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_discard_skips_without_the_skip_warning(self, mock_build_wf):
        upscale = MagicMock(return_value=StageOutcome.success)
        mock_build_wf.return_value = self._workflow(upscale)
        skip = SkipSignal()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            events = self._run(skip, lambda _e: skip.set_review_decision("discard"))

        finished = next(event for event in events if event["type"] == "generation_finished")
        assert finished["status"] == "skipped"
        upscale.assert_not_called()

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_an_all_discarded_batch_completes(self, mock_build_wf):
        mock_build_wf.return_value = self._workflow(MagicMock(return_value=StageOutcome.success))
        skip = SkipSignal()

        events = self._run(skip, lambda _e: skip.set_review_decision("discard"), prompts=2)

        statuses = [event["status"] for event in events if event["type"] == "generation_finished"]
        assert statuses == ["skipped", "skipped"]
        assert events[-1]["type"] == "batch_completed"
        assert events[-1]["completed_iterations"] == 2

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_quit_during_a_review_skips_then_cancels_the_batch(self, mock_build_wf):
        upscale = MagicMock(return_value=StageOutcome.success)
        mock_build_wf.return_value = self._workflow(upscale)
        skip = SkipSignal()

        events = self._run(skip, lambda _e: skip.queue_action("quit"), prompts=2)

        types = [event["type"] for event in events]
        finished = [event for event in events if event["type"] == "generation_finished"]
        assert [event["status"] for event in finished] == ["skipped"]
        assert types[-2:] == ["generation_finished", "batch_cancelled"]
        assert "batch_completed" not in types
        assert types.count("job_paused") == 1
        upscale.assert_not_called()

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_the_wait_is_not_generation_time(self, mock_build_wf):
        import time

        mock_build_wf.return_value = self._workflow(MagicMock(return_value=StageOutcome.success))
        skip = SkipSignal()

        def _decide_late(_event):
            time.sleep(0.3)
            skip.set_review_decision("approve")

        events = self._run(skip, _decide_late, prompts=2)

        finished = [event for event in events if event["type"] == "generation_finished"]
        assert all(event["generation_time"] < 0.2 for event in finished)
        second_start = [event for event in events if event["type"] == "prompt_started"][1]
        assert second_start["avg_secs"] < 0.2

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_a_pause_queued_during_the_small_render_applies_after_the_upscale(self, mock_build_wf):
        import threading
        import time

        upscale = MagicMock(return_value=StageOutcome.success)
        mock_build_wf.return_value = self._workflow(upscale)
        skip = SkipSignal()

        def _decide(_event):
            skip.queue_action("pause")
            skip.set_review_decision("approve")

        def _resume_a_pause(event):
            if event["type"] == "job_paused" and "review" not in event:
                assert upscale.call_count == 1

                def _resume():
                    while not skip.is_waiting_for_resume():
                        time.sleep(0.001)
                    skip.resume()

                threading.Thread(target=_resume, daemon=True).start()

        events = self._run(skip, _decide, on_event=_resume_a_pause)

        types = [event["type"] for event in events]
        assert types.count("job_paused") == 2
        assert "review" in events[types.index("job_paused")]
        assert types.index("generation_finished") < len(types) - 1 - types[::-1].index("job_paused")

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_a_pause_queued_before_the_review_applies_after_a_discard(self, mock_build_wf):
        import threading
        import time

        upscale = MagicMock(return_value=StageOutcome.success)
        mock_build_wf.return_value = self._workflow(upscale)
        skip = SkipSignal()

        def _decide(_event):
            skip.queue_action("pause")
            skip.set_review_decision("discard")

        def _resume_a_pause(event):
            if event["type"] == "job_paused" and "review" not in event:

                def _resume():
                    while not skip.is_waiting_for_resume():
                        time.sleep(0.001)
                    skip.resume()

                threading.Thread(target=_resume, daemon=True).start()

        events = self._run(skip, _decide, on_event=_resume_a_pause)

        types = [event["type"] for event in events]
        assert types.count("job_paused") == 2
        assert "review" in events[types.index("job_paused")]
        plain_pause = len(types) - 1 - types[::-1].index("job_paused")
        assert types.index("generation_finished") < plain_pause
        assert types[plain_pause + 1] == "job_resumed"
        assert skip.pending() is None
        upscale.assert_not_called()
        assert types[-1] == "batch_completed"

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_a_quit_while_paused_after_a_discard_ends_the_batch(self, mock_build_wf):
        import threading
        import time

        mock_build_wf.return_value = self._workflow(MagicMock(return_value=StageOutcome.success))
        skip = SkipSignal()

        def _decide(_event):
            skip.queue_action("pause")
            skip.set_review_decision("discard")

        def _quit_the_pause(event):
            if event["type"] == "job_paused" and "review" not in event:

                def _quit():
                    while not skip.is_waiting_for_resume():
                        time.sleep(0.001)
                    skip.queue_action("quit")
                    skip.resume()

                threading.Thread(target=_quit, daemon=True).start()

        events = self._run(skip, _decide, on_event=_quit_the_pause, prompts=2)

        types = [event["type"] for event in events]
        assert types[-1] == "batch_cancelled"
        assert types.count("prompt_started") == 1

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_a_repeat_queued_before_the_review_applies_after_a_discard(self, mock_build_wf):
        upscale = MagicMock(return_value=StageOutcome.success)
        mock_build_wf.return_value = self._workflow(upscale)
        skip = SkipSignal()
        decisions = iter(["discard", "approve"])

        def _decide(_event):
            decision = next(decisions)
            if decision == "discard":
                skip.queue_action("repeat")
            skip.set_review_decision(decision)

        events = self._run(skip, _decide)

        types = [event["type"] for event in events]
        assert types.count("job_paused") == 2
        assert "generation_repeat" in types
        assert types.index("generation_finished") < types.index("generation_repeat")
        statuses = [event["status"] for event in events if event["type"] == "generation_finished"]
        assert statuses == ["skipped", "success"]
        upscale.assert_called_once()
        assert types[-1] == "batch_completed"

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_a_next_queued_before_the_review_skips_the_image_without_an_upscale(self, mock_build_wf):
        upscale = MagicMock(return_value=StageOutcome.success)
        mock_build_wf.return_value = self._workflow(upscale)
        skip = SkipSignal()

        def _queue_next_while_rendering(event):
            if event["type"] == "generation_started":
                skip.queue_action("skip")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            events = self._run(skip, lambda _e: None, on_event=_queue_next_while_rendering, prompts=2)

        finished = [event["status"] for event in events if event["type"] == "generation_finished"]
        assert finished == ["skipped", "skipped"]
        assert [event["type"] for event in events].count("generation_started") == 2
        upscale.assert_not_called()
        assert events[-1]["type"] == "batch_completed"

    @pytest.mark.parametrize("args", [dict(upscale=2, review=False), dict(upscale=None, review=True)])
    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_no_review_callable_without_upscale_and_review(self, mock_build_wf, args):
        stage = MagicMock(return_value=StageOutcome.success)
        mock_build_wf.return_value = GenerationWorkflow(name="test", stages=[stage])

        run_batch(MagicMock(), MagicMock(spec=[]), _prompts(), _CONFIG, _make_args(**args), model_info=_MODEL_INFO, plan=_make_plan(_prompts(), _make_args(**args), config=_CONFIG))

        assert stage.call_args[0][0].review is None

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_review_callable_is_set_with_upscale_and_review(self, mock_build_wf):
        stage = MagicMock(return_value=StageOutcome.success)
        mock_build_wf.return_value = GenerationWorkflow(name="test", stages=[stage])
        args = _make_args(upscale=2, review=True)

        run_batch(
            MagicMock(),
            MagicMock(spec=[]),
            _prompts(),
            _CONFIG,
            args,
            model_info=_MODEL_INFO,
            plan=_make_plan(_prompts(), args, config=_CONFIG),
            skip_signal=SkipSignal(),
        )

        assert callable(stage.call_args[0][0].review)

    @patch("zvisiongenerator.image_runner.build_workflow")
    def test_a_review_discard_still_warns_for_other_skips(self, mock_build_wf):
        """Only a decision made in a review silences the skip warning."""
        mock_build_wf.return_value = _mock_workflow(StageOutcome.skipped)

        with pytest.warns(UserWarning, match="Generation skipped by workflow stage"):
            run_batch(
                MagicMock(),
                MagicMock(spec=[]),
                _prompts(),
                _CONFIG,
                _make_args(upscale=2, review=True),
                model_info=_MODEL_INFO,
                plan=_make_plan(_prompts(), _make_args(upscale=2, review=True), config=_CONFIG),
                skip_signal=SkipSignal(),
            )
