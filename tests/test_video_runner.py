"""Tests for video batch orchestration (run_video_batch)."""

from __future__ import annotations

from conftest import _make_plan, make_extend_source

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import _make_mock_video_backend, _make_video_args
from zvisiongenerator.core.job_plan import EnhanceStatus, JobPlan
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import ExtendSpec
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings
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

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args, plan=_make_plan(prompts, args, config=config))
        assert workflow.run.call_count == 1

    def test_multiple_prompts_multiple_runs(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success
        args = _make_video_args(runs=2)
        prompts = {"set1": [("prompt A", None), ("prompt B", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args, plan=_make_plan(prompts, args, config=config))
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

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args, plan=_make_plan(prompts, args, config=config))
        assert workflow.run.call_count == 3

    def test_zero_prompts_early_return(self):
        backend = _make_mock_video_backend()
        model = MagicMock()
        workflow = MagicMock()
        events: list[dict[str, object]] = []
        args = _make_video_args(runs=1)
        prompts: dict = {}
        config: dict = {}

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args, progress_callback=events.append, plan=_make_plan(prompts, args, config=config))
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

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args, progress_callback=events.append, plan=_make_plan(prompts, args, config=config))

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
                "message": "Video generation failed.",
            }
        ]
        assert len(batch_events) == 1
        assert batch_events[0]["message"] == "Video generation failed."
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

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args, progress_callback=events.append, plan=_make_plan(prompts, args, config=config))

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

        run_video_batch(backend, model, _ltx_model_info(), workflow, prompts, config, args, progress_callback=events.append, plan=_make_plan(prompts, args, config=config))

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


@pytest.mark.filterwarnings("ignore:Video generation failed")
class TestVideoFailureMessages:
    """Failed generations and a failed batch carry a one-line reason."""

    @staticmethod
    def _run(workflow, *, runs=1, prompts=None):
        events: list[dict[str, object]] = []
        args = _make_video_args(runs=runs)
        prompts = prompts or {"default": [("p", None)]}
        config: dict = {}
        run_video_batch(
            _make_mock_video_backend(),
            MagicMock(),
            _ltx_model_info(),
            workflow,
            prompts,
            config,
            args,
            progress_callback=events.append,
            plan=_make_plan(prompts, args, config=config),
        )
        return events

    def test_exception_message_is_one_line_on_finished_and_batch_failed(self):
        workflow = MagicMock()
        workflow.run.side_effect = RuntimeError("CUDA out of memory\n  while allocating\ttensor")

        events = self._run(workflow)

        (finished,) = [e for e in events if e["type"] == "generation_finished"]
        (batch,) = [e for e in events if e["type"] == "batch_failed"]
        assert finished["status"] == "failed"
        assert finished["message"] == "CUDA out of memory while allocating tensor"
        assert batch["message"] == finished["message"]

    def test_exception_without_text_falls_back_to_its_type_name(self):
        workflow = MagicMock()
        workflow.run.side_effect = ValueError()

        events = self._run(workflow)

        (finished,) = [e for e in events if e["type"] == "generation_finished"]
        assert finished["message"] == "ValueError"

    def test_stage_failure_uses_the_message_the_stage_set(self):
        def _fail(request, artifacts):
            artifacts.failure_message = "Extend produced no video segment."
            return StageOutcome.failed

        workflow = MagicMock()
        workflow.run.side_effect = _fail

        events = self._run(workflow)

        (finished,) = [e for e in events if e["type"] == "generation_finished"]
        (batch,) = [e for e in events if e["type"] == "batch_failed"]
        assert finished["message"] == "Extend produced no video segment."
        assert batch["message"] == "Extend produced no video segment."

    def test_batch_failed_carries_the_last_failed_generations_message(self):
        messages = iter(["first reason", "second reason"])

        def _fail(request, artifacts):
            artifacts.failure_message = next(messages)
            return StageOutcome.failed

        workflow = MagicMock()
        workflow.run.side_effect = _fail

        events = self._run(workflow, runs=2)

        finished = [e for e in events if e["type"] == "generation_finished"]
        (batch,) = [e for e in events if e["type"] == "batch_failed"]
        assert [e["message"] for e in finished] == ["first reason", "second reason"]
        assert batch["message"] == "second reason"

    def test_success_and_completed_batch_have_no_message(self):
        def _succeed(request, artifacts):
            artifacts.filename = "clip.mp4"
            return StageOutcome.success

        workflow = MagicMock()
        workflow.run.side_effect = _succeed

        events = self._run(workflow)

        for event in events:
            if event["type"] in {"generation_finished", "batch_completed"}:
                assert "message" not in event


class TestVideoSceneRequests:
    """The runner hands each iteration's planned scene (and the job's recorded inputs) to its request."""

    def _requests(self, prompts, *, plan_scene=None, scene_by_set=None, plan_transform=None, **arg_overrides):
        captured: list = []
        workflow = MagicMock()
        workflow.run.side_effect = lambda request, artifacts: captured.append(request) or StageOutcome.success
        args = _make_video_args(runs=1, **arg_overrides)
        plan = _make_plan(prompts, args, config={}, scene=plan_scene, scene_by_set=scene_by_set)
        if plan_transform is not None:
            plan = plan_transform(plan)
        run_video_batch(_make_mock_video_backend(), MagicMock(), _ltx_model_info(), workflow, prompts, {}, args, plan=plan)
        return captured

    def test_plain_job_has_no_scene_fields(self):
        (request,) = self._requests({"s": [("a sunset", None)]})
        assert (request.scene, request.resolved_scene, request.enhanced_scene, request.scene_input, request.prompt_input) == (None,) * 5

    def test_job_scene_reaches_every_request(self):
        requests = self._requests({"s": [("a", None), ("b", None)]}, plan_scene="A {castle|castle}.")
        assert [(r.scene, r.resolved_scene) for r in requests] == [("A {castle|castle}.", "A castle.")] * 2

    def test_entry_scene_beats_the_job_scene(self):
        requests = self._requests({"s": [("a", None), ("b", None)]}, plan_scene="Job.", scene_by_set={"s": ["Entry.", None]})
        assert [r.scene for r in requests] == ["Entry.", "Job."]

    def test_scene_only_job_runs_with_an_empty_prompt(self):
        (request,) = self._requests({"prompt": [("", None)]}, plan_scene="A castle at dusk.")
        assert request.prompt == "" and request.resolved_prompt == "" and request.resolved_scene == "A castle at dusk."

    def test_enhanced_pair_reaches_the_request(self):
        def _enhance(plan):
            return JobPlan(
                iterations=tuple(
                    replace(it, enhance=EnhanceSettings(), enhanced_prompt="A knight rides in.", enhanced_scene="A grey castle.", enhance_status=EnhanceStatus.ENHANCED) for it in plan.iterations
                )
            )

        (request,) = self._requests({"s": [("a knight", None)]}, plan_scene="A castle.", plan_transform=_enhance)
        assert (request.enhanced_prompt, request.enhanced_scene) == ("A knight rides in.", "A grey castle.")
        assert request.scene == "A castle." and request.prompt == "a knight"

    def test_recorded_inputs_come_from_args(self):
        (request,) = self._requests({"s": [("A knight rides in.", None)]}, scene_input="A castle.", prompt_input="a knight rides in")
        assert request.scene_input == "A castle." and request.prompt_input == "a knight rides in"


class TestVideoSceneEndToEnd:
    """Run the real stages behind ``run_video_batch`` and check the one composed prompt sent and the config recorded."""

    def _run(self, tmp_path, prompts, *, enhance=False, plan_transform=None, scene=None, scene_by_set=None):
        from argparse import Namespace
        from unittest.mock import patch

        from zvisiongenerator.workflows import build_video_workflow

        backend = _make_mock_video_backend()
        backend.text_to_video.return_value = tmp_path / "out.mp4"
        args = _make_video_args(runs=1, output=str(tmp_path))
        plan = _make_plan(prompts, args, config={}, scene=scene, scene_by_set=scene_by_set)
        if plan_transform is not None:
            plan = plan_transform(plan)
        workflow = build_video_workflow(Namespace(image_path=None, no_audio=False), enhance=enhance)
        with patch("zvisiongenerator.workflows.video_stages.embed_mp4_config") as embed:
            run_video_batch(backend, MagicMock(), _ltx_model_info(), workflow, prompts, {}, args, plan=plan)
        sent = [call.kwargs["prompt"] for call in backend.text_to_video.call_args_list]
        return sent, [call.args[1] for call in embed.call_args_list]

    def test_entry_scene_beats_job_scene_and_is_sent_once(self, tmp_path):
        sent, payloads = self._run(tmp_path, {"s": [("a knight rides in", None), ("a dog runs", None)]}, scene="Job.", scene_by_set={"s": ["Entry.", None]})
        assert sent == ["Entry. a knight rides in", "Job. a dog runs"]
        assert [p["scene"] for p in payloads] == ["Entry.", "Job."]
        assert [p["prompt"] for p in payloads] == ["a knight rides in", "a dog runs"]

    def test_enhanced_pair_is_sent_once_and_recorded_apart(self, tmp_path):
        def _enhance(plan):
            return JobPlan(
                iterations=tuple(
                    replace(it, enhance=EnhanceSettings(), enhanced_prompt="A knight gallops in.", enhanced_scene="A grey castle.", enhance_status=EnhanceStatus.ENHANCED) for it in plan.iterations
                )
            )

        sent, (payload,) = self._run(tmp_path, {"s": [("a knight", None)]}, enhance=True, plan_transform=_enhance, scene="A castle.")
        assert sent == ["A grey castle. A knight gallops in."]
        assert sent[0].count("A grey castle.") == 1 and "A castle." not in sent[0]
        assert (payload["scene"], payload["prompt"]) == ("A grey castle.", "A knight gallops in.")
        assert (payload["scene_input"], payload["prompt_input"]) == ("A castle.", "a knight")

    def test_plain_run_records_no_scene_keys(self, tmp_path):
        sent, (payload,) = self._run(tmp_path, {"s": [("a sunset", None)]})
        assert sent == ["a sunset"]
        assert not {"scene", "scene_input", "prompt_input"} & payload.keys()


class TestVideoQuit:
    """Verify a queued Quit stops a video batch before the next video or mid-video."""

    @staticmethod
    def _run(workflow, *, runs: int, skip_signal):
        events: list[dict[str, object]] = []
        args = _make_video_args(runs=runs, steps=8)
        prompts = {"default": [("a sunset", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 100}}
        run_video_batch(
            _make_mock_video_backend(),
            MagicMock(),
            _ltx_model_info(),
            workflow,
            prompts,
            config,
            args,
            progress_callback=events.append,
            plan=_make_plan(prompts, args, config=config),
            skip_signal=skip_signal,
        )
        return events

    def test_quit_queued_before_a_video_stops_the_batch_without_running_it(self):
        skip_signal = SkipSignal()
        skip_signal.queue_action("quit")
        workflow = MagicMock()

        events = self._run(workflow, runs=2, skip_signal=skip_signal)

        workflow.run.assert_not_called()
        assert events[-1] == {"type": "batch_cancelled", "mode": "video", "completed_iterations": 0, "total_iterations": 2}

    def test_quit_stops_the_running_video_at_its_next_step_and_skips_the_rest(self):
        skip_signal = SkipSignal()
        steps_seen: list[int] = []

        def _denoise(request, artifacts):
            for step in range(1, 9):
                if step == 3:
                    skip_signal.queue_action("quit")
                request.step_callback({"current_step": step, "total_steps": 8})
                steps_seen.append(step)
            return StageOutcome.success

        workflow = MagicMock()
        workflow.run.side_effect = _denoise

        events = self._run(workflow, runs=3, skip_signal=skip_signal)

        assert steps_seen == [1, 2]
        assert workflow.run.call_count == 1
        assert not [event for event in events if event["type"] == "generation_finished"]
        assert events[-1]["type"] == "batch_cancelled"
        assert events[-1]["completed_iterations"] == 0

    def test_other_actions_are_ignored_for_video(self):
        skip_signal = SkipSignal()
        skip_signal.queue_action("pause")
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success

        events = self._run(workflow, runs=2, skip_signal=skip_signal)

        assert workflow.run.call_count == 2
        assert events[-1]["type"] == "batch_completed"


class TestEndImageBatch:
    """An end image in the arguments reaches each run's request."""

    @pytest.mark.parametrize(("end_image", "expected"), [("/tmp/last.png", "/tmp/last.png"), (None, None)])
    def test_request_carries_the_end_image(self, end_image, expected):
        requests = []
        workflow = MagicMock()
        workflow.run.side_effect = lambda request, artifacts: requests.append(request) or StageOutcome.success
        args = _make_video_args(image_path="/tmp/first.png", end_image_path=end_image)
        prompts = {"default": [("a dog", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 10**9}}

        run_video_batch(_make_mock_video_backend(), MagicMock(), _ltx_model_info(), workflow, prompts, config, args, plan=_make_plan(prompts, args, config=config))

        (request,) = requests
        assert (request.image_path, request.end_image_path) == ("/tmp/first.png", expected)


class TestExtendBatch:
    """Verify an extension job builds its request from the extend arguments and plans each run with its own seed."""

    def _source(self):
        return make_extend_source(scene="A street at night.", prompt="She walks away.")

    def _run(self, *, runs=1, prompts=None, **arg_overrides):
        requests = []
        workflow = MagicMock()
        workflow.run.side_effect = lambda request, artifacts: requests.append(request) or StageOutcome.success
        source = self._source()
        args = _make_video_args(runs=runs, width=512, height=256, num_frames=25 + 48, extend_source=source, extend_frames=48, **arg_overrides)
        prompts = prompts or {"default": [("", None)]}
        config: dict = {"generation": {"seed_min": 4, "seed_max": 10**9}}
        run_video_batch(_make_mock_video_backend(), MagicMock(), _ltx_model_info(), workflow, prompts, config, args, plan=_make_plan(prompts, args, config=config, scene=source.scene))
        return requests, source

    def test_request_carries_the_extend_spec_and_the_window_size(self):
        (request,), source = self._run()
        assert request.extend == ExtendSpec(source, 48)
        assert (request.width, request.height, request.num_frames) == (512, 256, 73)
        assert request.image_path is None and request.upscale is None

    def test_planned_scene_reaches_the_request_and_an_empty_prompt_is_accepted(self):
        (request,), _ = self._run()
        assert request.prompt == "" and request.scene == "A street at night." and request.resolved_scene == "A street at night."

    def test_prints_the_added_frames_and_the_source_name(self, capsys):
        self._run()
        assert "+48 frames from clip.mp4" in capsys.readouterr().out

    def test_sibling_runs_get_different_seeds(self):
        with patch("zvisiongenerator.preflight.random.randint", side_effect=[1111, 2222]):
            requests, _ = self._run(runs=2, seed=None)
        assert [r.seed for r in requests] == [1111, 2222]
        assert all(r.extend is not None and r.extend.frames == 48 for r in requests)

    def test_fixed_seed_repeats_across_runs(self):
        requests, _ = self._run(runs=2, seed=9)
        assert [r.seed for r in requests] == [9, 9]

    def test_the_request_records_the_picked_model_name_over_the_resolved_model(self):
        (request,), _ = self._run(model="org/resolved-repo", model_name="ltx-4")
        assert request.model_name == "ltx-4"

    def test_other_jobs_have_no_extend(self):
        workflow = MagicMock()
        workflow.run.return_value = StageOutcome.success
        args = _make_video_args()
        prompts = {"default": [("a sunset", None)]}
        run_video_batch(_make_mock_video_backend(), MagicMock(), _ltx_model_info(), workflow, prompts, {}, args, plan=_make_plan(prompts, args))
        assert workflow.run.call_args.args[0].extend is None
