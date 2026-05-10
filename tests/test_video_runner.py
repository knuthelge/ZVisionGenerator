"""Tests for video batch orchestration (run_video_batch)."""

from __future__ import annotations

from unittest.mock import MagicMock

from tests.conftest import _make_mock_video_backend, _make_video_args
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.video_runner import run_video_batch


def _ltx_model_info():
    return VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)


class TestRunVideoBatch:
    """Verify run_video_batch() loop execution and edge cases."""

    def test_single_prompt_single_run(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success
        args = _make_video_args(runs=1)
        prompts = {"default": [("a sunset", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args)
        assert workflow.run.call_count == 1

    def test_multiple_prompts_multiple_runs(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success
        args = _make_video_args(runs=2)
        prompts = {"set1": [("prompt A", None), ("prompt B", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args)
        # 2 runs * 2 prompts = 4 iterations
        assert workflow.run.call_count == 4

    def test_multiple_sets(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success
        args = _make_video_args(runs=1)
        prompts = {"set_a": [("p1", None)], "set_b": [("p2", None), ("p3", None)]}
        config: dict = {}

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args)
        assert workflow.run.call_count == 3

    def test_zero_prompts_early_return(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        events: list[dict[str, object]] = []
        args = _make_video_args(runs=1)
        prompts: dict = {}
        config: dict = {}

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args, progress_callback=events.append)
        workflow.run.assert_not_called()

        assert events == [
            {
                "type": "batch_completed",
                "mode": "video",
                "total_iterations": 0,
                "completed_iterations": 0,
            }
        ]

    def test_failed_generation_emits_failed_terminal_batch_event(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.failed
        events: list[dict[str, object]] = []
        args = _make_video_args(runs=1)
        prompts = {"default": [("fail prompt", None)]}
        config: dict = {}

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args, progress_callback=events.append)

        request = workflow.run.call_args.args[0]

        finished_events = [event for event in events if event["type"] == "generation_finished"]
        batch_events = [event for event in events if event["type"] == "batch_failed"]

        assert request.backend is backend
        assert request.model is model
        assert request.prompt == "fail prompt"
        assert request.seed == args.seed

        assert finished_events == [
            {
                "type": "generation_finished",
                "mode": "video",
                "run_index": 0,
                "total_runs": 1,
                "ran_iterations": 1,
                "total_iterations": 1,
                "set_name": "default",
                "prompt_index": 0,
                "total_prompts": 1,
                "prompt": "fail prompt",
                "seed": args.seed,
                "status": "failed",
            }
        ]
        assert len(batch_events) == 1
        assert not [event for event in events if event["type"] == "batch_completed"]
        assert batch_events[0]["mode"] == "video"
        assert batch_events[0]["completed_iterations"] == 1
        assert batch_events[0]["total_iterations"] == 1
        assert isinstance(batch_events[0]["total_time"], float)

    def test_step_progress_includes_video_generation_context(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        events: list[dict[str, object]] = []
        captured_requests = []

        def _emit_step_then_succeed(request, artifacts):
            captured_requests.append(request)
            assert request.step_callback is not None
            request.step_callback({"current_step": 3, "total_steps": request.steps, "phase": "video_generate"})
            artifacts.generation_time = 0.25
            artifacts.filename = "clip.mp4"
            return StageOutcome.success

        workflow = MagicMock()
        workflow.run.side_effect = _emit_step_then_succeed
        args = _make_video_args(runs=1, steps=8)
        prompts = {"default": [("a sunset", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args, progress_callback=events.append)

        step_events = [event for event in events if event["type"] == "step_progress"]

        assert len(captured_requests) == 1
        assert captured_requests[0].backend is backend
        assert captured_requests[0].model is model
        assert captured_requests[0].prompt == "a sunset"
        assert captured_requests[0].seed == args.seed

        assert len(step_events) == 1
        assert step_events[0]["mode"] == "video"
        assert step_events[0]["phase"] == "video_generate"
        assert step_events[0]["current_step"] == 3
        assert step_events[0]["total_steps"] == 8
        assert step_events[0]["run_index"] == 0
        assert step_events[0]["total_runs"] == 1
        assert step_events[0]["ran_iterations"] == 1
        assert step_events[0]["total_iterations"] == 1
        assert step_events[0]["set_name"] == "default"
        assert step_events[0]["prompt_index"] == 0
        assert step_events[0]["total_prompts"] == 1

    def test_success_emits_generation_finished_with_shared_payload(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        events: list[dict[str, object]] = []
        captured_requests = []

        def _succeed_with_artifacts(request, artifacts):
            captured_requests.append(request)
            artifacts.generation_time = 0.25
            artifacts.filename = "clip.mp4"
            artifacts.video_path = "/tmp/clip.mp4"
            return StageOutcome.success

        workflow = MagicMock()
        workflow.run.side_effect = _succeed_with_artifacts
        args = _make_video_args(runs=1, steps=8, seed=77)
        prompts = {"default": [("a sunset", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args, progress_callback=events.append)

        finished_events = [event for event in events if event["type"] == "generation_finished"]

        assert len(captured_requests) == 1
        assert captured_requests[0].backend is backend
        assert captured_requests[0].model is model
        assert captured_requests[0].prompt == "a sunset"
        assert captured_requests[0].seed == 77

        assert len(finished_events) == 1
        assert finished_events[0]["mode"] == "video"
        assert finished_events[0]["status"] == "success"
        assert finished_events[0]["run_index"] == 0
        assert finished_events[0]["total_runs"] == 1
        assert finished_events[0]["ran_iterations"] == 1
        assert finished_events[0]["total_iterations"] == 1
        assert finished_events[0]["set_name"] == "default"
        assert finished_events[0]["prompt_index"] == 0
        assert finished_events[0]["total_prompts"] == 1
        assert finished_events[0]["prompt"] == "a sunset"
        assert finished_events[0]["seed"] == 77
        assert captured_requests[0].prompt == finished_events[0]["prompt"]
        assert captured_requests[0].seed == finished_events[0]["seed"]
        assert finished_events[0]["filename"] == "clip.mp4"
        assert finished_events[0]["output_path"] == "/tmp/clip.mp4"
        assert finished_events[0]["generation_time"] == 0.25
